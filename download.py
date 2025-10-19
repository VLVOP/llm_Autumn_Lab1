from datasets import load_dataset

print ("downloading iwslt2017(en-de)...")

dataset = load_dataset("iwslt2017", "iwslt2017-en-de")

print("complete...")
print (dataset)