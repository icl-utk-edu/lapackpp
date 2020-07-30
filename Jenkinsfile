pipeline {

agent none
triggers { cron ('H H(0-2) * * *') }
stages {
    //======================================================================
    stage('Parallel Build') {
        matrix {
            axes {
                axis {
                    name 'maker'
                    values 'make'
                }
                axis {
                    name 'host'
                    values 'master'
                }
            } // axes
            stages {
                stage('Build') {
                    agent { label 'lips' }

                    //----------------------------------------------------------
                    steps {
                        sh '''
                        #!/bin/sh +x
                        hostname && pwd

                        source /home/jenkins/spack_setup
                        sload gcc@6.4.0
                        sload intel-mkl
                        sload netlib-lapack

                        echo "========================================"
                        echo "maker ${maker}"
                        if [ "${maker}" = "make" ]; then
                            export color=no
                            make distclean
                            make config CXXFLAGS="-Werror"
                            export top=..

                            # Modify make.inc to add netlib LAPACKE for bug fixes.
                            ##export LAPACKDIR=/home/jmfinney/projects/lapack/build/lib
                            #export LAPACKDIR=/var/lib/jenkins/workspace/jmfinney/netlib-xylitol/build/lib
                            export LAPACKDIR=`spack location -i netlib-lapack`
                            sed -i -e 's/LIBS *=/LIBS = -L${LAPACKDIR}\/lib64 -llapacke /' make.inc
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
                        changed {
                            slackSend channel: '#slate_ci',
                                color: 'good',
                                message: "${currentBuild.fullDisplayName} >> ${STAGE_NAME} >> ${maker} ${host} changed (<${env.BUILD_URL}|Open>)"
                        }
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "${currentBuild.fullDisplayName} >> ${STAGE_NAME} >> ${maker} ${host} unstable (<${env.BUILD_URL}|Open>)"
                        }
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "${currentBuild.fullDisplayName} >> ${STAGE_NAME} >> ${maker} ${host} failed (<${env.BUILD_URL}|Open>)"
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
