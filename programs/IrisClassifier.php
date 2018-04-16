<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\MCC;
use Rubix\Engine\Tests\F1Score;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\KNearestNeighbors;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Tests\Informedness;
use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use League\Csv\Reader;

$k = $argv[1] ?? 3;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Classifier using K Nearest Neighbors           ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')->setDelimiter(',');

$dataset = SupervisedDataset::fromIterator($dataset);

list ($training, $testing) = $dataset->randomize()->stratifiedSplit(0.2);

$prototype = new Prototype(new KNearestNeighbors(3, new Euclidean()), [new Accuracy(), new F1Score(), new MCC(), new Informedness()]);

$prototype->train($training);

$prototype->test($testing);
