pipeline {
    agent none
    triggers { cron ('H H(0-2) * * *') }
    stages {
        //======================================================================
        stage('Parallel Build') {
            parallel {
                //--------------------------------------------------------------
                stage('Build - Master gcc MKL') {
                    agent { label 'master' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "LAPACK++ Building"
                        hostname && pwd

                        source /opt/spack/share/spack/setup-env.sh
                        spack load gcc
                        spack load intel-mkl

                        export color=no
                        make config
                        #make config CXXFLAGS="-Werror"

                        # Modify make.inc to add netlib LAPACKE for bug fixes.
                        export LAPACKDIR=/var/lib/jenkins/workspace/jmfinney/netlib-xylitol/build/lib
                        sed -i -e 's/LIBS *=/LIBS = -L${LAPACKDIR} -llapacke /' make.inc

                        make -j4
                        ldd test/tester
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "Master build - ${currentBuild.fullDisplayName} unstable: (<${env.BUILD_URL}|Open>)"
                        }
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "Master build - ${currentBuild.fullDisplayName} failed: (<${env.BUILD_URL}|Open>)"
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Build Failure",
                                body: "Master build - See more at ${env.BUILD_URL}"
                        }
                    } // post
                } // stage(Build - Master)

                //--------------------------------------------------------------
                stage('Build - Lips gcc CUDA MKL'){
                    agent { node 'lips.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "LAPACK++ Building"
                        hostname && pwd

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load cuda
                        spack load intel-mkl

                        export color=no
                        make config
                        #make config CXXFLAGS="-Werror"

                        # Modify make.inc to add netlib LAPACKE for bug fixes.
                        export LAPACKDIR=/home/jmfinney/projects/lapack/build/lib
                        sed -i -e 's/LIBS *=/LIBS = -L${LAPACKDIR} -llapacke /' make.inc

                        make -j4
                        ldd test/tester
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "Lips build - ${currentBuild.fullDisplayName} unstable: (<${env.BUILD_URL}|Open>)"
                        }
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "Lips build - ${currentBuild.fullDisplayName} failed: (<${env.BUILD_URL}|Open>)"
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Build Failure",
                                body: "Master build - See more at ${env.BUILD_URL}"
                        }
                    } // post
                } // stage(Build - Lips)
            } // parallel
        } // stage(Parallel Build)

        //======================================================================
        stage('Parallel Test') {
            parallel {
                //--------------------------------------------------------------
                stage('Test - Master') {
                    agent { label 'master' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "LAPACK++ Testing"
                        hostname && pwd

                        source /opt/spack/share/spack/setup-env.sh
                        spack load gcc
                        spack load intel-mkl

                        cd test
                        ./run_tests.py --ref n --xml report.xml
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "Master test - ${currentBuild.fullDisplayName} unstable: (<${env.BUILD_URL}|Open>)"
                        }
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "Master test - ${currentBuild.fullDisplayName} failed: (<${env.BUILD_URL}|Open>)"
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Build Failure",
                                body: "Master build - See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit 'test/*.xml'
                        }
                    } // post
                } // stage(Test - Master)

                //--------------------------------------------------------------
                stage('Test - Lips') {
                    agent { node 'lips.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "LAPACK++ Testing"
                        hostname && pwd

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load cuda
                        spack load intel-mkl

                        cd test
                        ./run_tests.py --ref n --xml report.xml
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "Lips test - ${currentBuild.fullDisplayName} unstable: (<${env.BUILD_URL}|Open>)"
                        }
                        // Lips currently has spurious errors; don't email them.
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "Lips test - ${currentBuild.fullDisplayName} failed: (<${env.BUILD_URL}|Open>)"
                        }
                        always {
                            junit 'test/*.xml'
                        }
                    } // post
                } // stage(Test - Lips)
            } // parallel
        } // stage(Parallel Test)
    } // stages
} // pipeline
