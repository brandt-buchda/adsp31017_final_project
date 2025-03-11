# ADSP 31017 Final Project

## Contributors

Richard Wang

Gavi Xiang

Cambell Taylor

Katy Koo

Brandt Buchda

## Required Dependecies

```shell
pip install numpy
pip install pandas
pip install unidecode
pip install pickle
pip install scikit-learn
```

## Data Schema

| Field        | Type         | Example                                                   |
|--------------|--------------|-----------------------------------------------------------|
| title        | String       | Inception                                                 |
| release_year | Integer      | 2010                                                      |
| box_office   | Float        | 829000000                                                 |
| budget       | Float        | 160000000                                                 |
| rating       | String       | pg-13                                                     |
| collection   | String       |                                                           |
| cast         | List[String] | Leonardo DiCaprio, Joseph Gordon-Levitt                   |
| director     | String       | Christopher Nolan                                         |
| writers      | List[String] | Christopher Nolan, Jonathan Nolan                         |
| distributors | List[String] | Warner Bros., Legendary Pictures                          |
| genres       | List[String] | Sci-Fi, Thriller                                          |
| plot         | String       | A thief enters dreams to steal secrets and plant ideas... |


## Generating Predictions

Run the following command to generate predictions


```shell
python3 main.py --csv data.csv --test
```

## Prediction Results

Label 0: \$0-\$1M

Label 1: \$1M-\$10M

Label 2: \$10M-\$50M

Label 3: \$50M-\$100M

Label 4: \$100M-\$200M

Label 5: \$200M-\$500M