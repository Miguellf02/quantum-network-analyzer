pipeline {

    agent any

    environment {
        VENV = "${WORKSPACE}/venv"
    }

    stages {

        stage('Checkout') {
            steps {
                echo "Checking out repository..."
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo "🐍 Creating Python virtual env..."

                sh """
                    set -e

                    python3 --version

                    # Crear entorno virtual
                    python3 -m venv ${VENV}

                    # Activar entorno
                    . ${VENV}/bin/activate

                    # Asegurar pip actualizado
                    python -m pip install --upgrade pip

                    # Instalar dependencias
                    if [ -f requirements.txt ]; then
                        echo 'Installing from requirements.txt...'
                        python -m pip install -r requirements.txt
                    else
                        echo '⚠ No requirements.txt found, installing basics...'
                        python -m pip install numpy pandas scikit-learn matplotlib seaborn
                    fi
                """
            }
        }

        stage('Run QKD Pipeline') {
            steps {
                echo " Running QKD Analysis Pipeline..."

                sh """
                    set -e
                    . ${VENV}/bin/activate
                    python src/main.py
                """
            }
        }

        stage('Validate Outputs') {
            steps {
                echo " Validating output artifacts..."

                sh """
                    test -f data/processed/python_preprocessing/QKD_PROCESSED.csv
                    test -f data/processed/feature_engineered/QKD_FEATURES.csv
                    echo 'Critical artifacts found ✔'
                """
            }
        }
    }

    post {

        always {
            echo "ℹ The pipeline has finished its execution."
        }

        success {
            echo " Pipeline SUCCESS — archiving artifacts..."

            archiveArtifacts artifacts: '''
                data/processed/**/*.csv,
                data/processed/**/*.png
            ''', fingerprint: true

            echo " Artifacts archived and available for download."
        }

        failure {
            echo " Pipeline FAILED — check logs."
        }
    }
}
