pipeline {

agent none
triggers { pollSCM 'H/10 * * * *' }
stages {
    //======================================================================
    stage('Parallel Build') {
        matrix {
            axes {
                axis {
                    name 'maker'
                    values 'make', 'cmake'
                }
                axis {
                    name 'host'
                    values 'caffeine'
                }
            } // axes
            stages {
                stage('Build') {
                    agent { node "${host}.icl.utk.edu" }

                    //----------------------------------------------------------
                    steps {
                        sh '''
#!/bin/sh +x
hostname && pwd
export top=`pwd`

source /home/jenkins/spack_setup
sload gcc@6.4.0
sload intel-mkl
#sload netlib-lapack

echo "========================================"
echo "maker ${maker}"
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    export color=no
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install

    # Used to need LAPACKE from Netlib, but MKL has been fixed.
    ## Modify make.inc to add netlib LAPACKE for bug fixes.
    #export LAPACKDIR=`spack location -i netlib-lapack`/lib64
    #sed -i -e 's/LIBS *=/LIBS = -L${LAPACKDIR} -llapacke /' make.inc
fi
if [ "${maker}" = "cmake" ]; then
    sload cmake

    rm -rf blaspp
    git clone https://bitbucket.org/icl/blaspp
    mkdir blaspp/build && cd blaspp/build
    cmake -Dcolor=no -Dbuild_tests=no -DCMAKE_INSTALL_PREFIX=${top}/install ..
    make -j8 install
    cd ../..

    rm -rf build && mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" -DCMAKE_INSTALL_PREFIX=${top}/install \
          -DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp ..
fi

echo "========================================"
make -j8
make -j8 install
ls -R ${top}/install

echo "========================================"
ldd test/tester

echo "========================================"
cd test
export OMP_NUM_THREADS=8
./run_tests.py --quick --xml ${top}/report-${maker}.xml

echo "========================================"
echo "Verify install with smoke tests."
cd ${top}/example

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH=${top}/install/lib/pkgconfig
    make clean
fi
if [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake "-DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp;${top}/install/lib64/lapackpp" ..
fi

make
./example_potrf || exit 1
'''
                    } // steps

                    //----------------------------------------------------------
                    post {
                        failure {
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} >> ${STAGE_NAME} >> ${maker} ${host} failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit '*.xml'
                        }
                    } // post

                } // stage(Build)
            } // stages
        } // matrix
    } // stage(Parallel Build)
} // stages

} // pipeline
