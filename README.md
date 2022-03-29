# CIDER-LM

## Python Virtual Environment

- `conda env create -f ./environment.yml`
- `conda activate melt`

## Execution

- 2 options:
  - `java -cp target/ciderlm-1.0.jar es.unizar.sid.ciderlm.Main`
  - `mvn exec:java -Dexec.mainClass="es.unizar.sid.ciderlm.Main"`
