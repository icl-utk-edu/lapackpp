pipeline {

agent none
triggers { cron ('H H(4-5) * * *') }
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
                    values 'master'
                }
            } // axes
            stages {
                stage('Build') {
                    agent { label 'master' }

                    //----------------------------------------------------------
                    steps {
                        sh '''
                        #!/bin/sh +x
                        hostname && pwd
                        export top=`pwd`

                        source /home/jenkins/spack_setup
                        sload gcc@6.4.0
                        sload intel-mkl
                        sload netlib-lapack

                        #source /opt/spack/share/spack/setup-env.sh
                        #spack load gcc
                        #spack load intel-mkl

                        echo "========================================"
                        echo "maker ${maker}"
                        if [ "${maker}" = "make" ]; then
                            export color=no
                            make distclean
                            make config CXXFLAGS="-Werror"

                            # Modify make.inc to add netlib LAPACKE for bug fixes.
                            export LAPACKDIR=`spack location -i netlib-lapack`/lib64
                            sed -i -e 's/LIBS *=/LIBS = -L${LAPACKDIR} -llapacke /' make.inc
                        fi
                        if [ "${maker}" = "cmake" ]; then
                            sload cmake

                            ls -R

                            rm -rf blaspp
                            git clone https://bitbucket.org/icl/blaspp
                            mkdir blaspp/build
                            cd blaspp/build
                            cmake -Dcolor=no -Dbuild_tests=no ..
                            make -j4 lib
                            cd ../..

                            rm -rf build
                            mkdir build
                            cd build
                            cmake -Dcolor=no -Dblaspp_DIR=${top}/blaspp/build -DCMAKE_CXX_FLAGS="-Werror" ..
                        fi

                        echo "========================================"
                        make -j4

                        echo "========================================"
                        ldd test/tester

                        echo "========================================"
                        cd test
                        ./run_tests.py --small --xml ${top}/report-${maker}.xml
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
