import re
import config
from   config import print_header, print_subhead, print_line, print_result
from   config import get

#-------------------------------------------------------------------------------
def blas():
    print_header( 'BLAS library' )
    print( 'Also detects Fortran name mangling and BLAS integer size.' )

    test_mkl        = (config.environ['mkl']        == '1')
    test_acml       = (config.environ['acml']       == '1')
    test_essl       = (config.environ['essl']       == '1')
    test_openblas   = (config.environ['openblas']   == '1')
    test_accelerate = (config.environ['accelerate'] == '1')
    # otherwise, test all
    test_all = not (test_mkl or test_acml or test_essl or test_openblas or
                    test_accelerate)

    # build list of choices to test
    choices = []

    if (test_all):
        # sometimes BLAS is in default libraries (e.g., on Cray)
        choices.extend([
            ('Default', {'LIBS': ''}),
        ])
    # end

    if (test_all or test_mkl):
        choices.extend([
            # each pair has Intel conventions, then GNU conventions
            # int, threaded
            ('Intel MKL (int, Intel conventions)',
                {'LIBS':     '-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp',
                 'LDFLAGS':  '-fopenmp'}),
            ('Intel MKL (int, GNU conventions)',
                {'LIBS':     '-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp',
                 'LDFLAGS':  '-fopenmp'}),

            # int64_t, threaded
            ('Intel MKL (int64_t, Intel conventions)',
                {'LIBS':     '-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp -DMKL_ILP64',
                 'LDFLAGS':  '-fopenmp'}),
            ('Intel MKL (int64_t, GNU conventions)',
                {'LIBS':     '-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp -DMKL_ILP64',
                 'LDFLAGS':  '-fopenmp'}),

            # int, sequential
            ('Intel MKL (int, Intel conventions)',
                {'LIBS':     '-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': ''}),
            ('Intel MKL (int, GNU conventions)',
                {'LIBS':     '-lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': ''}),

            # int64_t, sequential
            ('Intel MKL (int64_t, Intel conventions)',
                {'LIBS':     '-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': '-DMKL_ILP64'}),
            ('Intel MKL (int64_t, GNU conventions)',
                {'LIBS':     '-lmkl_gf_ilp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': '-DMKL_ILP64'}),
        ])
    # end

    if (test_all or test_acml):
        choices.extend([
            ('AMD ACML', {'LIBS': '-lacml'}),
            ('AMD ACML', {'LIBS': '-lacml_mp'}),
        ])
    # end

    if (test_all or test_essl):
        choices.extend([
            ('IBM ESSL', {'LIBS': '-lessl'}),
        ])
    # end

    if (test_all or test_openblas):
        choices.extend([
            ('OpenBLAS', {'LIBS': '-lopenblas'}),
        ])
    # end

    if (test_all or test_accelerate):
        choices.extend([
            ('MacOS Accelerate', {'LIBS': '-framework Accelerate'}),
        ])
    # end

    passed = []
    for (label, env) in choices:
        title = label
        if (env['LIBS']):
            title += '\n    ' + env['LIBS']
        print_subhead( title )
        # BLAS uses the FORTRAN_*; LAPACK uses older ADD_, NOCHANGE, UPCASE.
        for mangling in ('-DFORTRAN_ADD_  -DADD_',
                         '-DFORTRAN_LOWER -DNOCHANGE',
                         '-DFORTRAN_UPPER -DUPCASE'):
            for size in ('', '-DBLAS_ILP64'):
                print_line( '    ' + mangling +' '+ size )
                env['CXXFLAGS'] = get(env, 'CXXFLAGS') +' '+ mangling +' '+ size
                config.environ.push()
                config.environ.merge( env )
                (rc, out, err) = config.compile_exe( 'config/blas.cc' )
                config.environ.pop()
                # if int32 didn't link, int64 won't either
                if (rc != 0):
                    print_result( label, rc )
                    break

                # if int32 runs, skip int64
                (rc, out2, err2) = config.run( 'config/blas' )
                print_result( label, rc )
                if (rc == 0):
                    break
            # end
            # break on first mangling that works
            if (rc == 0):
                break
        # end
        if (rc == 0):
            passed.append( (label, env) )
            if (config.auto):
                break
    # end

    labels = map( lambda c: c[0] + ': ' + c[1]['LIBS'], passed )
    i = config.choose( labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', '-DHAVE_BLAS' )
# end blas

#-------------------------------------------------------------------------------
def cblas():
    print_header( 'CBLAS library' )
    choices = [
        ('Default, in BLAS library', {'LIBS': ''}),
        ('Netlib CBLAS: -lcblas',    {'LIBS': '-lcblas'}),
    ]

    passed = []
    for (label, env) in choices:
        config.environ.push()
        config.environ.merge( env )
        (rc, out, err) = config.compile_run( 'config/cblas.cc', label )
        config.environ.pop()
        if (rc == 0):
            passed.append( (label, env) )
            break
    # end

    labels = map( lambda c: c[0] + ': ' + c[1]['LIBS'], passed )
    i = config.choose( labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', '-DHAVE_CBLAS' )
# end lapacke

#-------------------------------------------------------------------------------
# Should -llapack be appended or prepended?
# Depends: -llapack -lopenblas (LAPACK requires BLAS),
# but:     -lessl -llapack (LAPACK provides functions missing in ESSL).
# Safest is prepend (what autoconf does), but then LAPACK may override
# optimized versions in ESSL. config.environ.merge prepends LIBS.
def lapack():
    print_header( 'LAPACK library' )
    choices = [
        ('Default, in BLAS library', {'LIBS': ''}),
        ('Netlib LAPACK: -llapack',  {'LIBS': '-llapack'}),
    ]

    passed = []
    for (label, env) in choices:
        config.environ.push()
        config.environ.merge( env )
        (rc, out, err) = config.compile_run( 'config/lapack.cc', label )
        config.environ.pop()
        if (rc == 0):
            passed.append( (label, env) )
            break
    # end

    labels = map( lambda c: c[0] + ': ' + c[1]['LIBS'], passed )
    i = config.choose( labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', '-DHAVE_LAPACK' )
# end lapack

#-------------------------------------------------------------------------------
def lapacke():
    print_header( 'LAPACKE library' )
    choices = [
        ('Default, in LAPACK library', {'LIBS': ''}),
        ('Netlib LAPACKE: -llapacke',  {'LIBS': '-llapacke'}),
    ]

    passed = []
    for (label, env) in choices:
        config.environ.push()
        config.environ.merge( env )
        (rc, out, err) = config.compile_run( 'config/lapacke.cc', label )
        config.environ.pop()
        if (rc == 0):
            passed.append( (label, env) )
            break
    # end

    labels = map( lambda c: c[0] + ': ' + c[1]['LIBS'], passed )
    i = config.choose( labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', '-DHAVE_LAPACKE' )
# end lapacke

#-------------------------------------------------------------------------------
def blas_float_return():
    (rc, out, err) = config.compile_run(
        'config/return_float.cc',
        'BLAS returns float as float (standard)' )
    if (rc == 0):
        return

    (rc, out, err) = config.compile_run(
        'config/return_float_f2c.cc',
        'BLAS returns float as double (f2c convention)' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', '-DHAVE_F2C' )
    else:
        print( ansi_bold + ansi_red + 'unexpected error!' + ansi_normal )
# end

#-------------------------------------------------------------------------------
def blas_complex_return():
    (rc, out, err) = config.compile_run(
        'config/return_complex.cc',
        'BLAS returns complex (GNU gfortran convention)' )
    if (rc == 0):
        return

    (rc, out, err) = config.compile_run(
        'config/return_complex_argument.cc',
        'BLAS returns complex as hidden argument (Intel ifort, f2c convention)' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', '-DBLAS_COMPLEX_RETURN_ARGUMENT' )
    else:
        print( ansi_bold + ansi_red + 'unexpected error!' + ansi_normal )
# end

#-------------------------------------------------------------------------------
def lapack_version():
    config.print_line( 'LAPACK version' )
    (rc, out, err) = config.compile_run( 'config/lapack_version.cc' )
    s = re.search( r'^LAPACK_VERSION=((\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        v = '%d%02d%02d' % (int(s.group(2)), int(s.group(3)), int(s.group(4)))
        config.environ.append( 'CXXFLAGS', '-DLAPACK_VERSION=%s' % v )
        config.print_result( 'LAPACK', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'LAPACK', rc )
# end

#-------------------------------------------------------------------------------
def lapack_xblas():
    (rc, out, err) = config.compile_run( 'config/xblas.cc', 'LAPACK XBLAS' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', '-DHAVE_XBLAS' )
# end

#-------------------------------------------------------------------------------
def lapack_matgen():
    (rc, out, err) = config.compile_run( 'config/matgen.cc', 'LAPACK MATGEN' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', '-DHAVE_MATGEN' )
# end

#-------------------------------------------------------------------------------
def mkl_version():
    config.print_line( 'MKL version' )
    (rc, out, err) = config.compile_run( 'config/mkl_version.cc' )
    s = re.search( r'^MKL_VERSION=((\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        config.environ.append( 'CXXFLAGS', '-DHAVE_MKL' )
        config.print_result( 'MKL', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'MKL', rc )
# end

#-------------------------------------------------------------------------------
def essl_version():
    config.print_line( 'ESSL version' )
    (rc, out, err) = config.compile_run( 'config/essl_version.cc' )
    s = re.search( r'^ESSL_VERSION=((\d+)\.(\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        config.environ.append( 'CXXFLAGS', '-DHAVE_ESSL' )
        config.print_result( 'ESSL', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'ESSL', rc )
# end

#-------------------------------------------------------------------------------
def openblas_version():
    config.print_line( 'OpenBLAS version' )
    (rc, out, err) = config.compile_run( 'config/openblas_version.cc' )
    s = re.search( r'^OPENBLAS_VERSION=((\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        config.environ.append( 'CXXFLAGS', '-DHAVE_OPENBLAS' )
        config.print_result( 'OpenBLAS', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'OpenBLAS', rc )
# end
