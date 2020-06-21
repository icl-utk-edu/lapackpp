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
            } // axes
            stages {
                stage('Build') {
                    agent { label 'master' }

                    //----------------------------------------------------------
                    steps {
                        sh '''
                        #!/bin/sh +x
                        hostname && pwd

                        # LAPACK++ currently has spurious errors on lips,
                        # so run only on master.
                        # on lips
                        #source /home/jmfinney/spack/share/spack/setup-env.sh
                        #spack load gcc@6.4.0

                        # on master
                        source /opt/spack/share/spack/setup-env.sh
                        spack load gcc

                        spack load intel-mkl

                        echo "========================================"
                        echo "maker ${maker}"
                        if [ "${maker}" = "make" ]; then
                            export color=no
                            make distclean
                            make config CXXFLAGS="-Werror"
                            export top=..

                            # Modify make.inc to add netlib LAPACKE for bug fixes.
                            # on lips
                            #export LAPACKDIR=/home/jmfinney/projects/lapack/build/lib
                            export LAPACKDIR=/var/lib/jenkins/workspace/jmfinney/netlib-xylitol/build/lib
                            sed -i -e 's/LIBS *=/LIBS = -L${LAPACKDIR} -llapacke /' make.inc
                        fi

                        echo "========================================"
                        # master is 4 core
                        make -j4

                        echo "========================================"
                        ldd test/tester

                        echo "========================================"
                        cd test
                        ./run_tests.py --xml ${top}/report-${maker}.xml
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
