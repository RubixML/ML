<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\CART;
use Rubix\Engine\Math\Random;
use Rubix\Engine\Math\Stats;
use Rubix\Engine\Transformers\RandomSplitter;
use League\Csv\Reader;

$minSize = $argv[1] ?? 3;
$maxDepth = $argv[2] ?? PHP_INT_MAX;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Classifier using CART model                    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')->setDelimiter(',');
$dataset = new RandomSplitter(iterator_to_array($dataset), 0.2);

$estimator = new CART($minSize, $maxDepth, ['Sepal Width', 'Sepal Height', 'Petal Width', 'Petal Height', 'Classification']);

echo 'Training a CART ... ';

$start = microtime(true);

$estimator->train($dataset->training());

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Testing model ...';

$start = microtime(true);

$accuracy = $estimator->test($dataset->testing());

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Model is ' . (string) Stats::round($accuracy, 5) . ' accurate.' . "\n";

echo  "\n";

echo 'Random Sample Input' . "\n";

$sample = [
    Random::float(4.0, 8.0),
    Random::float(2.0, 4.0),
    Random::float(1.0, 7.0),
    Random::float(0.0, 3.0),
];

echo 'Sepal size: ' . $sample[0] . 'cm X ' . $sample[1] . 'cm' . "\n";
echo 'Petal size: ' . $sample[2] . 'cm X ' . $sample[3] . 'cm' . "\n";

echo  "\n";

echo 'Making prediction ... ';

$start = microtime(true);

$prediction = $estimator->predict($sample);

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Label: ' . $prediction['label'] . "\n";
echo 'Outcome: ' . $prediction['outcome'] . "\n";
echo 'Certainty: ' . $prediction['certainty'] . "\n";
