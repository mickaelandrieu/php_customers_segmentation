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

$dataset = Unlabeled::fromIterator(new CSV('./data/customers.csv', true))
    ->apply(new NumericStringConverter())
;

$kmeans = PersistentModel::load(new Filesystem('./models/customers_clustering.rbx'));

$predictions = $kmeans->predict($dataset);

$variables = new ColumnPicker(new CSV('./data/customers.csv', true), [
    'id_customer',
    'recency',
    'monetary',
    'frequency'
]);

$ids = array_column(iterator_to_array($variables), 'id_customer');
$recencies = array_column(iterator_to_array($variables), 'recency');
$monetary_values = array_column(iterator_to_array($variables), 'monetary');
$frequencies = array_column(iterator_to_array($variables), 'frequency');

array_unshift($ids, 'id_customer');
array_unshift($recencies, 'recency');
array_unshift($monetary_values, 'monetary');
array_unshift($frequencies, 'frequency');
array_unshift($predictions, 'cluster');

$predictionsFile = new CSV('./output/predictions.csv');

$predictionsFile->export(array_map(null, ...[$ids, $predictions, $recencies, $frequencies, $monetary_values]));
