### Cityscapes

get the dataset here: 
https://opendatalab.com/OpenDataLab/CityScapes/tree/main
create an account and then download with the cli. (--dataset-repo argument does not work anymore.)
see: https://opendatalab.com/OpenDataLab/CityScapes/cli/main


openxlab dataset info --dataset-repo OpenDataLab/CityScapes # Dataset information viewing and View Dataset File List

openxlab dataset get --dataset-repo OpenDataLab/CityScapes #Dataset download

openxlab dataset download --dataset-repo OpenDataLab/CityScapes --source-path /README.md --target-path /path/to/local/folder #Dataset file download