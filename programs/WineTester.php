<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Test;
use Rubix\Engine\Math\Stats;
use Rubix\Engine\Math\Random;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\SupervisedDataset;
use League\Csv\Reader;

$epochs = $argv[1] ?? 500;
$batchSize = $argv[2] ?? 5;
$rate = $argv[3] ?? 1.0;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Tester using a Multi Layer Perceptron          ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')->setDelimiter(',')->getRecords();

$dataset = new SupervisedDataset(iterator_to_array($dataset));

list ($training, $testing) = $dataset->randomize()->split(0.5);

$estimator = new MultiLayerPerceptron(11, [10, 10, 10], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], $epochs, $batchSize, $rate);

echo 'Training a Multi Layer Perceptron ... ';

$start = microtime(true);

$estimator->train($training->samples(), $training->outcomes());

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

$test = new Test($estimator);

echo 'Testing model ...';

$start = microtime(true);

$accuracy = $test->accuracy($testing->samples(), $testing->outcomes());

echo  "\n";

echo 'Model is ' . (string) $accuracy . '% accurate.' . "\n";

echo  "\n";

echo 'Random Sample Input' . "\n";

$sample = [
    Random::float(4.6, 15.9, 2),
    Random::float(0.12, 1.58, 2),
    Random::float(0, 1, 2),
    Random::float(0.9, 15.5, 2),
    Random::float(0.012, 0.61, 3),
    Random::float(1, 72, 2),
    Random::float(6, 289, 0),
    Random::float(0.99007, 1.00369, 5),
    Random::float(2.74, 4.01, 2),
    Random::float(0.33, 2, 2),
    Random::float(8.4, 14.9, 2),
];

echo 'Fixed Acidity: ' . $sample[0] . "\n";
echo 'Volatile Acidity: ' . $sample[1] . "\n";
echo 'Ciric Acid: ' . $sample[2] . "\n";
echo 'Residual Sugar: ' . $sample[3] . "\n";
echo 'Chlorides: ' . $sample[4] . "\n";
echo 'Free Sulphur Dioxide: ' . $sample[5] . "\n";
echo 'Total Sulphur Dioxide: ' . $sample[6] . "\n";
echo 'Density: ' . $sample[7] . "\n";
echo 'pH: ' . $sample[8] . "\n";
echo 'Sulphates: ' . $sample[9] . "\n";
echo 'Alcohol: ' . $sample[10] . "\n";

echo  "\n";

echo 'Making prediction ... ';

$start = microtime(true);

$prediction = $estimator->predict($sample);

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Outcome: ' . $prediction['outcome'] . "\n";
echo 'Activation: ' . $prediction['activation'] . "\n";
