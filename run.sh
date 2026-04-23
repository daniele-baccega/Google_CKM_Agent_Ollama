#!/bin/bash

for i in {1..20}
do
  cat InputDiscorsivi/Input${i}.txt | llm-rag/bin/adk run . > CasiDiscorsivi/Caso${i}.txt
done

for i in {1..20}
do
  cat InputTabellari/Input${i}.txt | llm-rag/bin/adk run . > CasiTabellari/Caso${i}.txt
done