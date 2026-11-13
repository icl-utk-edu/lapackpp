#!/usr/bin/env python3
"""Check independent BLAS/LAPACK suffixes with controlled link fixtures.

Run with CXX set to the desired compiler. These checks need no BLAS library;
they test symbol selection, not numerical correctness or vendor-header ABIs.
"""

import argparse
import contextlib
import io
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    cxx = shlex.split(os.environ.get('CXX', 'c++'))
    sys.path.insert(0, str(root))
    # The Make configuration module parses argv when imported.
    with patch.object(sys, 'argv', [sys.argv[0]]):
        import config
        import config.config
        import config.lapack
    config.config.namespace_ = 'LAPACK'
    config.config.log = io.StringIO()

    def run(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)

    cases = [
        ('plain', '', None),
        ('inherit', '64_', None),
        ('none', '64_', ''),
        ('reverse', '', '64_'),
        ('different', 'blas_', 'lapack_'),
        ('bundled-make', '64_', '64_'),
    ]
    with tempfile.TemporaryDirectory(dir=args.work_dir) as tmp:
        work = Path(tmp)
        # Test CMake's cache handling without configuring a vendor library.
        # The mocked link/run results isolate repeated includes and the finder
        # resetting its definitions, as happens after an option change.
        cache_check = work / 'check-cache.cmake'
        cache_check.write_text(r"""
cmake_minimum_required(VERSION 3.17)
function(try_run run_result compile_result)
    set(${run_result} 0 PARENT_SCOPE)
    set(${compile_result} TRUE PARENT_SCOPE)
    set(run_output "LAPACK_VERSION=3.12.00 ok" PARENT_SCOPE)
    get_property(count GLOBAL PROPERTY probes)
    math(EXPR count "${count} + 1")
    set_property(GLOBAL PROPERTY probes ${count})
endfunction()
set_property(GLOBAL PROPERTY probes 0)
set(LAPACK_LIBRARIES fixture)
set(blaspp_defines -DBLAS_ILP64)
set(lapack_symbol_suffix_def -DLAPACK_FORTRAN_SUFFIX=)
set(lapackpp_defs_ -DLAPACK_HAVE_LAPACK ${lapack_symbol_suffix_def})
include(cmake/LAPACKConfig.cmake)
get_property(first GLOBAL PROPERTY probes)
include(cmake/LAPACKConfig.cmake)
get_property(second GLOBAL PROPERTY probes)
if(NOT first EQUAL second)
    message(FATAL_ERROR "unchanged settings repeated the probes")
endif()
# The finder may rerun with different options but select the same library.
set(lapackpp_defs_ -DLAPACK_HAVE_LAPACK ${lapack_symbol_suffix_def})
include(cmake/LAPACKConfig.cmake)
if(NOT "-DLAPACK_VERSION=31200" IN_LIST lapackpp_defs_)
    message(FATAL_ERROR "cached probes lost the LAPACK version definition")
endif()
set(lapack_symbol_suffix_def -DLAPACK_FORTRAN_SUFFIX=64_)
list(FILTER lapackpp_defs_ EXCLUDE REGEX "^-DLAPACK_FORTRAN_SUFFIX=")
list(APPEND lapackpp_defs_ ${lapack_symbol_suffix_def})
get_property(before GLOBAL PROPERTY probes)
include(cmake/LAPACKConfig.cmake)
get_property(after GLOBAL PROPERTY probes)
if(NOT after GREATER before)
    message(FATAL_ERROR "changed suffix did not repeat the probes")
endif()
""")
        subprocess.check_output(['cmake', '-P', str(cache_check)],
                                cwd=root, text=True, stderr=subprocess.STDOUT)
        print('CMake probe cache: unchanged, reset, and changed suffix passed',
              flush=True)
        for name, blas_suffix, lapack_suffix in cases:
            include = work / name
            for library in ('blas', 'lapack'):
                (include / library).mkdir(parents=True)
                (include / library / 'defines.h').write_text('')
                # Preprocessing the tester only needs include boundaries.
                (include / library / 'util.hh').write_text('')
            effective = blas_suffix if lapack_suffix is None else lapack_suffix
            blas_flags = ['-DBLAS_ILP64', '-DBLAS_FORTRAN_ADD_']
            if blas_suffix:
                blas_flags += ['-DBLAS_FORTRAN_SUFFIX=' + blas_suffix]
            flags = blas_flags.copy()
            if lapack_suffix is not None:
                flags += ['-DLAPACK_FORTRAN_SUFFIX=' + lapack_suffix]
            common = cxx + ['-std=c++11', '-I' + str(include),
                            '-I' + str(root / 'include')]
            (include / 'cblas.h').write_text('''
#include <stdint.h>
#define OPENBLAS_VERSION "fixture"
extern "C" double cblas_ddot(int64_t, const double*, int64_t, const double*, int64_t);
extern "C" const char* openblas_get_config();
''')
            (include / 'lapacke.h').write_text('''
#include <stdint.h>
typedef int64_t lapack_int;
#define LAPACK_COL_MAJOR 102
extern "C" lapack_int LAPACKE_dpstrf(int, char, lapack_int, double*, lapack_int,
                                    lapack_int*, lapack_int*, double);
''')
            fixture = include / 'fixture.cc'
            fixture.write_text('''
#include <stdint.h>
#include <stddef.h>
extern "C" {
double cblas_ddot@BLAS@(int64_t n, const double* x, int64_t ix,
                     const double* y, int64_t iy) {
    double result = 0;
    for (int64_t i = 0; i < n; ++i) result += x[i*ix]*y[i*iy];
    return result;
}
double ddot_@BLAS@(const int64_t* n, const double* x, const int64_t* ix,
                   const double* y, const int64_t* iy) {
    return cblas_ddot@BLAS@(*n, x, *ix, y, *iy);
}
const char* openblas_get_config@BLAS@() { return "fixture"; }
void dpstrf_@LAPACK@(const char*, const int64_t* n, double*, const int64_t*,
                   int64_t*, int64_t* rank, const double*, double*, int64_t* info,
                   size_t) { *rank = *n; *info = 0; }
int64_t LAPACKE_dpstrf@LAPACK@(int, char, int64_t n, double*, int64_t,
                            int64_t*, int64_t* rank, double) {
    *rank = n; return 0;
}
void dpotrf_@LAPACK@() {}
}
'''.replace('@BLAS@', blas_suffix).replace('@LAPACK@', effective))
            obj = include / 'fixture.o'
            run(common + ['-c', str(fixture), '-o', str(obj)])
            for probe in ('cblas.cc', 'openblas_version.cc', 'lapack_pstrf.cc',
                          'lapacke_pstrf.cc'):
                exe = include / probe.replace('.cc', '')
                run(common + flags + [str(root / 'config' / probe), str(obj),
                                       '-o', str(exe)])
                run([str(exe)])
            public = include / 'public.cc'
            public.write_text('''
#include "lapack/mangling.h"
extern "C" void LAPACK_GLOBAL(dpotrf, DPOTRF)();
int main() { LAPACK_GLOBAL(dpotrf, DPOTRF)(); }
''')
            exe = include / 'public'
            run(common + flags + [str(public), str(obj), '-o', str(exe)])
            run([str(exe)])
            # Exercise Make's option parsing and probe order with these same
            # link fixtures. Replace only the compiler launcher, so generated
            # files and objects stay in the temporary directory.
            environment = config.Environments()
            environment.stack = [{}, {
                'BLAS_LIBRARIES': str(obj), 'LAPACK_LIBRARIES': str(obj),
                'blas_int': 'int64', 'fortran_mangling': 'add_',
                'CXX': 'c++', 'CXX_actual': 'g++',
                'blas_symbol_suffix': ('' if name == 'bundled-make'
                                       else blas_suffix or 'none'),
                'lapack_symbol_suffix': ('none' if lapack_suffix == ''
                                         else lapack_suffix or ''),
            }]
            executables = {}

            def compile_probe(source, env=None, label=None):
                flags = environment['CXXFLAGS'] + ' ' + (env or {}).get('CXXFLAGS', '')
                exe = include / ('make-' + Path(source).stem)
                command = common + shlex.split(flags) + [str(root / source),
                                                         str(obj), '-o', str(exe)]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise AssertionError(result.stderr)
                executables[source] = exe
                return 0, result.stdout, result.stderr

            def run_probe(source):
                return 0, run([str(executables[source])]), ''

            def compile_run(source, env=None, label=None):
                compile_probe(source, env, label)
                return run_probe(source)

            with patch.object(config, 'environ', environment), \
                 patch.object(config, 'compile_exe', compile_probe), \
                 patch.object(config, 'run_exe', run_probe), \
                 patch.object(config, 'compile_run', compile_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                config.lapack.blas()
                config.lapack.cblas()
                config.lapack.lapack()
                config.lapack.lapacke()

            # Exercise the tester's actual mappings without requiring hundreds
            # of vendor declarations in the controlled headers.
            for source, symbol, suffix in (
                ('test/lapacke_wrappers.hh', 'LAPACKE_dgesv', effective),
                ('test/cblas_wrappers.cc', 'cblas_dgbmv', blas_suffix),
            ):
                output = run(common + flags + ['-E', '-P', '-x', 'c++',
                                               str(root / source)])
                expected = symbol + suffix
                if not re.search(r'\b' + re.escape(expected) + r'\s*\(', output):
                    raise AssertionError(name + ': missing ' + expected)
                if suffix and re.search(r'\b' + symbol + r'\s*\(', output):
                    raise AssertionError(name + ': unexpected plain ' + symbol)
            print(name + ': Make options, probes, public mangling, and tester mappings passed',
                  flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.output, file=sys.stderr)
        raise SystemExit(error.returncode)
