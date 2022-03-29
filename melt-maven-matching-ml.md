# MELT, Maven, matching-ml

## Maven

- `mvn archetype:generate -DgroupId=es.unizar.sid.ciderlm -DartifactId=ciderlm -DarchetypeArtifactId=maven-archetype-quickstart -DarchetypeVersion=1.4 -DinteractiveMode=false`
- `mvn package`
- `mvn clean`

## MELT

- `git clone git@github.com:dwslab/melt.git`

### matching-ml Example: transformers

- [Example](https://github.com/dwslab/melt/tree/master/examples/transformers)

#### Virtual Environment + requirement libraries

- `conda env create -f environment.yml` in `/melt/examples/transformers/src/main/resources`
- `conda activate melt`

#### Execution

- `mvn package`
- `java -cp target/transformers-1.0-jar-with-dependencies.jar de.uni_mannheim.informatik.dws.melt.examples.transformers.Main -m zero -tracks conference -tm sentence-transformers/all-MiniLM-L6-v2`
- [Output](./melt-transformers-example.out.txt)
- [Results](../../melt/examples/transformers/results/results_2022-03-25_10-35-25/trackPerformanceCube.csv)

### es.unizar.sid.ciderlm

#### Maven project

- `mvn archetype:generate -DgroupId=es.unizar.sid.ciderlm -DartifactId=ciderlm -DarchetypeArtifactId=maven-archetype-quickstart -DarchetypeVersion=1.4 -DinteractiveMode=false`

#### Execution

- `java -cp target/ciderlm-1.0-SNAPSHOT.jar es.unizar.sid.ciderlm.Main`

- `mvn exec:java -Dexec.mainClass="es.unizar.sid.ciderlm.Main"`
