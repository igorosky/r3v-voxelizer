//! Runs the JavaScript test suite in `tests/js` against the JS bindings.
//!
//! The host cdylib of this test run is of no use to JavaScript, so the crate
//! is built for `wasm32-unknown-unknown` with wasm-pack (the build the README
//! documents: `--target web`), with `js` and the voxelizer features of this
//! test run. Every JS test case runs in its own Node.js process, so a trap
//! (e.g. a Rust panic) fails only that case.
//!
//! Needs `wasm-pack` (or `WASM_PACK`), the `wasm32-unknown-unknown` target
//! (`rustup target add wasm32-unknown-unknown`) and `node` (or `NODE`).
//!
//! With the `wgpu` feature the tests also need WebGPU, which Node.js lacks:
//! the `webgpu` package (Dawn) installed into `tests/js/node_modules` by
//! `npm ci` in `tests/js`, and a Vulkan device (a software one such as
//! lavapipe works).

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{self, Command},
    sync::Arc,
};

use libtest_mimic::{Arguments, Failed, Trial};

/// Name of the generated JS module (and of its `_bg.wasm` file).
const MODULE: &str = "r3v_voxelizer";

const TEST_SCRIPT: &str = "tests/js/voxelizer_tests.mjs";

/// Voxelizer features the module may be built with. The JS tests get the
/// enabled ones (comma separated, in `R3V_VOXELIZER_FEATURES`) and expect
/// exactly the matching classes. `multi_threaded` is not supported in JS
/// (build.rs refuses `js` together with it).
const VOXELIZER_FEATURES: [(&str, bool); 2] = [
    ("single_threaded", cfg!(feature = "single_threaded")),
    ("wgpu", cfg!(feature = "wgpu")),
];

/// Where the JS tests of the `wgpu` feature take WebGPU from.
const WEBGPU_PACKAGE: &str = "tests/js/node_modules/webgpu";

/// The generated JS package and the Node.js that runs the tests.
struct JsTests {
    node: OsString,
    script: PathBuf,
    pkg_dir: PathBuf,
}

impl JsTests {
    /// Builds the package into a directory unique to this process, so that
    /// concurrent test runs do not overwrite each other's package.
    fn setup() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(
            !cfg!(feature = "wgpu") || root.join(WEBGPU_PACKAGE).is_dir(),
            "{WEBGPU_PACKAGE} not found; install it with `npm ci` in tests/js, e.g. in docker: \
             docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v \"$PWD/tests/js:/w\" -w /w \
             node:24 npm ci"
        );
        let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
        let pkg_dir = tmp.join(format!("js-{}", process::id()));
        let features: Vec<&str> = ["js"].into_iter().chain(enabled_features()).collect();
        let wasm_pack = env::var_os("WASM_PACK").unwrap_or_else(|| "wasm-pack".into());
        // A target directory of its own: a different target than the test
        // run, and kept between runs so the wasm build stays incremental.
        let output = Command::new(&wasm_pack)
            .args(["build", "--dev", "--target", "web", "--no-pack"])
            .args(["--no-typescript", "--out-name", MODULE, "--out-dir"])
            .arg(&pkg_dir)
            .arg(root)
            .args(["--", "--no-default-features", "--features"])
            .arg(features.join(","))
            .env("CARGO_TARGET_DIR", tmp.join("js-wasm-target"))
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "failed to run `{}` (install wasm-pack or set WASM_PACK): {e}",
                    wasm_pack.to_string_lossy()
                )
            });
        assert!(
            output.status.success(),
            "building the wasm package failed ({}); it needs the \
             wasm32-unknown-unknown target:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        Self {
            node: env::var_os("NODE").unwrap_or_else(|| "node".into()),
            script: root.join(TEST_SCRIPT),
            pkg_dir,
        }
    }

    /// A command running the JS test script against the built package.
    fn command(&self) -> Command {
        let mut command = Command::new(&self.node);
        command
            .arg(&self.script)
            .env("R3V_VOXELIZER_PKG", &self.pkg_dir)
            .env("R3V_VOXELIZER_FEATURES", enabled_features().join(","));
        command
    }

    fn list_cases(&self) -> Vec<String> {
        let output = self.command().arg("list").output().unwrap_or_else(|e| {
            panic!(
                "failed to run Node.js `{}` (install it or set NODE): {e}",
                self.node.to_string_lossy()
            )
        });
        assert!(
            output.status.success(),
            "listing JS tests failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("JS test names are UTF-8")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn run_case(&self, name: &str) -> Result<(), Failed> {
        let output = self
            .command()
            .args(["run", name])
            .output()
            .map_err(|e| format!("failed to start JS test: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "{}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )
            .into())
        }
    }
}

fn enabled_features() -> Vec<&'static str> {
    VOXELIZER_FEATURES
        .iter()
        .filter(|(_, enabled)| *enabled)
        .map(|(feature, _)| *feature)
        .collect()
}

fn main() {
    let args = Arguments::from_args();
    let js_tests = Arc::new(JsTests::setup());
    let trials = js_tests
        .list_cases()
        .into_iter()
        .map(|name| {
            let js_tests = js_tests.clone();
            Trial::test(name.clone(), move || js_tests.run_case(&name))
        })
        .collect();
    let conclusion = libtest_mimic::run(&args, trials);
    let _ = fs::remove_dir_all(&js_tests.pkg_dir);
    conclusion.exit();
}
