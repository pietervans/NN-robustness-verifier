@ECHO OFF

for %%n in (fc1 fc2 fc3 fc4 fc5) do (
    for %%k in (0 1) do (
        ECHO Evaluating network net%%k_%%n
        for /f %%f in ('DIR %CD%\..\test_cases\net%%k_%%n\*.txt /b') do (
            python verifier.py --net net%%k_%%n --spec "%CD%\..\test_cases\net%%k_%%n\%%f"
        )
    )
)
