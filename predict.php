<?php

if (!($loader = @include __DIR__ . '/vendor/autoload.php')) {
    die(<<<EOT
You need to install the project dependencies using Composer:
$ wget http://getcomposer.org/composer.phar
OR
$ curl -s https://getcomposer.org/installer | php
$ php composer.phar install --dev
$ phpunit
EOT
    );
}

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\NumericStringConverter;

// We use the same dataset (it's a test, not a real ML app)
$dataset = Unlabeled::fromIterator(new CSV('./data/customers.csv', true))
    ->apply(new NumericStringConverter())
;

$variables = [
    'id_customer',
    'recency',
    'monetary',
    'frequency',
];

$kmeans = PersistentModel::load(new Filesystem('./models/customers_clustering.rbx'));

$predictions = $kmeans->predict($dataset);

// add the predictions variable to the former dataset
$selection = new ColumnPicker(new CSV('./data/customers.csv', true), $variables);

$valuesPerVariable = [];

foreach ($variables as $variable) {
    $variableValues = array_column(iterator_to_array($selection), $variable);
    array_unshift($variableValues, $variable);
    $valuesPerVariable[] = $variableValues;
}

// For further analysis : plot, statistics, testing...
$predictionsFile = new CSV('./output/predictions.csv');
$predictionsFile->export(array_map(null, ...$valuesPerVariable));
