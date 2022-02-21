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

use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;

// Import the Data
$dataset = Unlabeled::fromIterator(new CSV('./data/customers.csv', true))
    ->apply(new NumericStringConverter())
;

//dump($dataset->describe());
//dump($dataset->head());

// Scale and center the Data
// Same behavior than Scikit-Learn StandardScaler()
$dataset->apply(new ZScaleStandardizer());

// Set up the Machine Learning model : MiniBatch KMeans
// The first argument is the number of clusters we want to create

$kmeans = new PersistentModel(
    new KMeans(3),
    new Filesystem('./models/customers_clustering.rbx', true)
);

$kmeans->train($dataset);
$kmeans->save();
