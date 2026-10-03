//! Runs the C test suite in `tests/c` against the C bindings.
//!
//! The C sources are compiled against `bindings/r3v-voxelizer.h` and the
//! cdylib that cargo built for this test run. Every C test case runs in its
//! own process, so a crash (e.g. a NULL dereference) fails only that case.

use std::{
    env::{self, consts},
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{self, Command},
    sync::Arc,
};

use libtest_mimic::{Arguments, Failed, Trial};

const C_SOURCES: [&str; 2] = ["tests/c/harness.c", "tests/c/voxelizer_tests.c"];

/// Variable the dynamic loader searches before the paths baked into a binary.
const LIB_PATH_VAR: &str = if cfg!(target_os = "macos") {
    "DYLD_LIBRARY_PATH"
} else {
    "LD_LIBRARY_PATH"
};

/// The compiled C test program and the cdylib it must load.
struct CTests {
    exe: PathBuf,
    lib_dir: PathBuf,
}

impl CTests {
    /// Compiles the C tests into a file unique to this process, so that
    /// concurrent test runs do not overwrite each other's program.
    fn build() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let exe = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "c_tests-{}{}",
            process::id(),
            consts::EXE_SUFFIX
        ));
        let lib_dir = lib_dir();
        let cc = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
        let output = Command::new(&cc)
            .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-g"])
            .arg("-I")
            .arg(root.join("bindings"))
            .args(C_SOURCES.map(|src| root.join(src)))
            .arg("-o")
            .arg(&exe)
            .arg("-L")
            .arg(&lib_dir)
            .arg("-lr3v_voxelizer")
            .output()
            .unwrap_or_else(|e| panic!("failed to run C compiler `{cc}`: {e}"));
        assert!(
            output.status.success(),
            "compiling the C tests failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        Self { exe, lib_dir }
    }

    /// A command running the C test program against this build's cdylib.
    ///
    /// `cargo test` puts `target/<profile>` before `target/<profile>/deps`
    /// in the library path, and `target/<profile>/libr3v_voxelizer.so` is only
    /// refreshed by `cargo build`. Without putting `deps` first the program
    /// could load a stale copy (an rpath cannot help: the environment
    /// variable takes precedence over it).
    fn command(&self) -> Command {
        let mut paths = vec![self.lib_dir.clone()];
        if let Some(existing) = env::var_os(LIB_PATH_VAR) {
            paths.extend(env::split_paths(&existing));
        }
        let joined: OsString = env::join_paths(paths).expect("library path is joinable");
        let mut command = Command::new(&self.exe);
        command.env(LIB_PATH_VAR, joined);
        command
    }

    fn list_cases(&self) -> Vec<String> {
        let output = self
            .command()
            .arg("list")
            .output()
            .expect("failed to list C tests");
        assert!(
            output.status.success(),
            "listing C tests failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("C test names are UTF-8")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn run_case(&self, name: &str) -> Result<(), Failed> {
        let output = self
            .command()
            .args(["run", name])
            .output()
            .map_err(|e| format!("failed to start C test: {e}"))?;
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

/// `target/<profile>/deps`: where rustc writes the cdylib for this test run.
fn lib_dir() -> PathBuf {
    let lib_name = format!("{}r3v_voxelizer{}", consts::DLL_PREFIX, consts::DLL_SUFFIX);
    let exe = env::current_exe().expect("test binary path");
    let dir = exe.parent().expect("test binary has a parent directory");
    assert!(
        dir.join(&lib_name).is_file(),
        "{lib_name} not found next to {}; build with `--features c,multi_threaded`",
        exe.display()
    );
    dir.to_path_buf()
}

fn main() {
    let args = Arguments::from_args();
    let c_tests = Arc::new(CTests::build());
    let trials = c_tests
        .list_cases()
        .into_iter()
        .map(|name| {
            let c_tests = c_tests.clone();
            Trial::test(name.clone(), move || c_tests.run_case(&name))
        })
        .collect();
    let conclusion = libtest_mimic::run(&args, trials);
    let _ = fs::remove_file(&c_tests.exe);
    conclusion.exit();
}
