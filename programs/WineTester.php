<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Prototype;
use Rubix\Engine\RandomForest;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\RegressionAnalysis;
use League\Csv\Reader;

$trees = $argv[1] ?? 10;
$ratio = $argv[2] ?? 0.10;
$minSize = $argv[3] ?? 5;
$maxDepth = $argv[4] ?? 10;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Tester using a Random Forest                   ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')->setDelimiter(';')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new Prototype(new RandomForest($trees, $ratio, $minSize, $maxDepth), [
    new RegressionAnalysis(),
]);

$estimator->train($training);

$estimator->test($testing);
