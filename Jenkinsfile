pipeline {
    agent {
        docker {
            image 'python:3.10-slim'
            args '-u'   // evita problemas de buffering
        }
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python env') {
            steps {
                sh """
                    python3 -m venv venv
                    . venv/bin/activate

                    if [ -f requirements.txt ]; then
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    else
                        echo "No requirements.txt found — installing essential packages"
                        pip install pandas numpy scikit-learn matplotlib seaborn
                    fi
                """
            }
        }

        stage('Run QKD Pipeline') {
            steps {
                sh """
                    . venv/bin/activate
                    python src/main.py
                """
            }
        }

    }

    post {
        success {
            archiveArtifacts artifacts: '**/results/**, **/*.csv, **/*.pkl, **/logs/**', fingerprint: true
            echo "Pipeline ejecutada correctamente."
        }
        failure {
            echo "Falló la ejecución de la pipeline."
        }
    }
}
