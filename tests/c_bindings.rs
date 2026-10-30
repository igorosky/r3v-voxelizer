//! Runs the C test suite in `tests/c` against the C bindings and, with the
//! `cpp` feature, the C++ smoke tests in `tests/cpp` against the C++ header.
//!
//! The sources are compiled against `bindings/r3v-voxelizer.h` (or `.hpp`)
//! and the cdylib that cargo built for this test run. Every test case runs in
//! its own process, so a crash (e.g. a NULL dereference) fails only that case.
//! C++ cases are listed with a `cpp::` prefix.
//!
//! With the `wgpu` feature the tests are compiled with `R3V_TEST_WGPU` and
//! also run on the GPU, which needs a Vulkan device (a software one such as
//! lavapipe works).

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
const CPP_SOURCES: [&str; 1] = ["tests/cpp/voxelizer_tests.cpp"];

/// Variable the dynamic loader searches before the paths baked into a binary.
const LIB_PATH_VAR: &str = if cfg!(target_os = "macos") {
    "DYLD_LIBRARY_PATH"
} else {
    "LD_LIBRARY_PATH"
};

/// A compiled test program and the cdylib it must load.
struct TestProgram {
    exe: PathBuf,
    lib_dir: PathBuf,
}

/// How to compile one test program.
struct Language {
    /// Name of the program, and of the language in messages
    name: &'static str,
    /// Environment variable naming the compiler
    compiler_var: &'static str,
    /// Compiler used when `compiler_var` is not set
    default_compiler: &'static str,
    std: &'static str,
    sources: &'static [&'static str],
}

const C: Language = Language {
    name: "c",
    compiler_var: "CC",
    default_compiler: "cc",
    std: "-std=c11",
    sources: &C_SOURCES,
};

const CPP: Language = Language {
    name: "cpp",
    compiler_var: "CXX",
    default_compiler: "c++",
    std: "-std=c++17",
    sources: &CPP_SOURCES,
};

impl TestProgram {
    /// Compiles the tests of `language` into a file unique to this process,
    /// so that concurrent test runs do not overwrite each other's program.
    fn build(language: &Language) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let exe = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "{}_tests-{}{}",
            language.name,
            process::id(),
            consts::EXE_SUFFIX
        ));
        let lib_dir = lib_dir();
        let cc = env::var(language.compiler_var)
            .unwrap_or_else(|_| language.default_compiler.to_owned());
        let mut command = Command::new(&cc);
        if cfg!(feature = "wgpu") {
            command.arg("-DR3V_TEST_WGPU");
        }
        let output = command
            .args([language.std, "-Wall", "-Wextra", "-Werror", "-g"])
            .arg("-I")
            .arg(root.join("bindings"))
            .args(language.sources.iter().map(|src| root.join(src)))
            .arg("-o")
            .arg(&exe)
            .arg("-L")
            .arg(&lib_dir)
            .arg("-lr3v_voxelizer")
            .output()
            .unwrap_or_else(|e| panic!("failed to run {} compiler `{cc}`: {e}", language.name));
        assert!(
            output.status.success(),
            "compiling the {} tests failed ({}):\n{}",
            language.name,
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        Self { exe, lib_dir }
    }

    /// A command running the test program against this build's cdylib.
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
            .expect("failed to list tests");
        assert!(
            output.status.success(),
            "listing tests of {} failed ({}):\n{}",
            self.exe.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("test names are UTF-8")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn run_case(&self, name: &str) -> Result<(), Failed> {
        let output = self
            .command()
            .args(["run", name])
            .output()
            .map_err(|e| format!("failed to start test: {e}"))?;
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
    // With the prefix of their test names
    let mut programs = vec![("", Arc::new(TestProgram::build(&C)))];
    if cfg!(feature = "cpp") {
        programs.push(("cpp::", Arc::new(TestProgram::build(&CPP))));
    }
    let trials = programs
        .iter()
        .flat_map(|(prefix, program)| {
            program.list_cases().into_iter().map(move |name| {
                let program = program.clone();
                Trial::test(format!("{prefix}{name}"), move || program.run_case(&name))
            })
        })
        .collect();
    let conclusion = libtest_mimic::run(&args, trials);
    for (_, program) in &programs {
        let _ = fs::remove_file(&program.exe);
    }
    conclusion.exit();
}
