pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python environment') {
            steps {
                sh '''
                    python3 --version || true
                    python3 -m venv venv
                    . venv/bin/activate

                    if [ -f requirements.txt ]; then
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    else
                        echo "No requirements.txt found, installing basics"
                        pip install numpy pandas scikit-learn matplotlib seaborn
                    fi
                '''
            }
        }

        stage('Run QKD Pipeline') {
            steps {
                sh '''
                    . venv/bin/activate
                    python src/main.py
                '''
            }
        }

    }

    post {
        always {
            echo "Pipeline finished."
        }
        success {
            echo "Pipeline SUCCESS"
        }
        failure {
            echo "Pipeline FAILED"
        }
    }
}
