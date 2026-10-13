//! Runs the Python test suite in `tests/python` against the Python bindings.
//!
//! The cdylib that cargo built for this test run is copied, under the file
//! name of the extension module, into a temporary directory and imported from
//! there. Every Python test case runs in its own interpreter process, so a
//! crash fails only that case.
//!
//! With the `wgpu` feature the tests also run on the GPU, which needs a
//! Vulkan device (a software one such as lavapipe works).

use std::{
    env::{self, consts},
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{self, Command},
    sync::Arc,
};

use libtest_mimic::{Arguments, Failed, Trial};

/// Name of the cdylib and of the Python module it defines.
const MODULE: &str = "r3v_voxelizer";

const TEST_SCRIPT: &str = "tests/python/voxelizer_tests.py";

/// File name suffix under which Python imports an extension module.
const MODULE_SUFFIX: &str = if cfg!(windows) { ".pyd" } else { ".so" };

/// Voxelizer features the module may be built with. The Python tests get the
/// enabled ones (comma separated, in `R3V_VOXELIZER_FEATURES`) and expect
/// exactly the matching classes.
const VOXELIZER_FEATURES: [(&str, bool); 3] = [
    ("single_threaded", cfg!(feature = "single_threaded")),
    ("multi_threaded", cfg!(feature = "multi_threaded")),
    ("wgpu", cfg!(feature = "wgpu")),
];

/// The importable copy of the module and the interpreter that runs the tests.
struct PythonTests {
    python: OsString,
    script: PathBuf,
    module_dir: PathBuf,
}

impl PythonTests {
    /// Copies the cdylib into a directory unique to this process, so that
    /// concurrent test runs do not overwrite each other's module.
    fn setup() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let module_dir =
            Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("python-{}", process::id()));
        fs::create_dir_all(&module_dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", module_dir.display()));
        let lib = lib_path();
        let module = module_dir.join(format!("{MODULE}{MODULE_SUFFIX}"));
        fs::copy(&lib, &module).unwrap_or_else(|e| {
            panic!(
                "failed to copy {} to {}: {e}",
                lib.display(),
                module.display()
            )
        });
        Self {
            python: python(),
            script: root.join(TEST_SCRIPT),
            module_dir,
        }
    }

    /// A command running the Python test script with the module importable.
    fn command(&self) -> Command {
        let mut paths = vec![self.module_dir.clone()];
        if let Some(existing) = env::var_os("PYTHONPATH") {
            paths.extend(env::split_paths(&existing));
        }
        let joined: OsString = env::join_paths(paths).expect("PYTHONPATH is joinable");
        let features: Vec<&str> = VOXELIZER_FEATURES
            .iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(feature, _)| *feature)
            .collect();
        let mut command = Command::new(&self.python);
        // -B: no __pycache__ inside tests/python.
        command
            .arg("-B")
            .arg(&self.script)
            .env("PYTHONPATH", joined)
            .env("R3V_VOXELIZER_FEATURES", features.join(","));
        command
    }

    fn list_cases(&self) -> Vec<String> {
        let output = self.command().arg("list").output().unwrap_or_else(|e| {
            panic!(
                "failed to run Python interpreter `{}`: {e}",
                self.python.to_string_lossy()
            )
        });
        assert!(
            output.status.success(),
            "listing Python tests failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("Python test names are UTF-8")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn run_case(&self, name: &str) -> Result<(), Failed> {
        let output = self
            .command()
            .args(["run", name])
            .output()
            .map_err(|e| format!("failed to start Python test: {e}"))?;
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

/// `target/<profile>/deps/<lib>`: the cdylib rustc wrote for this test run.
/// The copy in `target/<profile>` is only refreshed by `cargo build`.
fn lib_path() -> PathBuf {
    let lib_name = format!("{}{MODULE}{}", consts::DLL_PREFIX, consts::DLL_SUFFIX);
    let exe = env::current_exe().expect("test binary path");
    let lib = exe
        .parent()
        .expect("test binary has a parent directory")
        .join(&lib_name);
    assert!(
        lib.is_file(),
        "{lib_name} not found next to {}; build with `--features python`",
        exe.display()
    );
    lib
}

/// The interpreter to run the tests with, found the way PyO3 finds the one
/// it builds for: `PYO3_PYTHON` when set, otherwise `python3` from `PATH`
/// (which is the active virtualenv's, if any).
fn python() -> OsString {
    env::var_os("PYO3_PYTHON")
        .unwrap_or_else(|| if cfg!(windows) { "python" } else { "python3" }.into())
}

fn main() {
    let args = Arguments::from_args();
    let python_tests = Arc::new(PythonTests::setup());
    let trials = python_tests
        .list_cases()
        .into_iter()
        .map(|name| {
            let python_tests = python_tests.clone();
            Trial::test(name.clone(), move || python_tests.run_case(&name))
        })
        .collect();
    let conclusion = libtest_mimic::run(&args, trials);
    let _ = fs::remove_dir_all(&python_tests.module_dir);
    conclusion.exit();
}
