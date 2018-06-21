<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Metrics\Distance\Euclidean;
use Rubix\ML\Metrics\Validation\F1Score;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\RobustStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Classifier using K Nearest Neighbors           ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
]));

$labels = iterator_to_array($reader->fetchColumn('class'));

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new KNearestNeighbors(3, new Euclidean()), [
    new NumericStringConverter(),
    new RobustStandardizer(),
]);

$validator = new KFold(10);

$report = new ConfusionMatrix();

var_dump($validator->test($estimator, $dataset, new F1Score()));

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));
