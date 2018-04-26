<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\DecisionForest;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Transformers\OneHotEncoder;
use Rubix\Engine\Metrics\Reports\ClassificationReport;
use League\Csv\Reader;

$trees = $argv[1] ?? 5;
$ratio = $argv[2] ?? 0.10;
$minSize = $argv[3] ?? 3;
$maxDepth = $argv[4] ?? 10;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using a Decision Forest         ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.3);

$estimator = new Prototype(new Pipeline(new DecisionForest($trees, $ratio, $minSize, $maxDepth), [
    new OneHotEncoder(),
]), [
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);
