<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\Speed;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\DecisionForest;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\OneHotEncoder;
use League\Csv\Reader;

$trees = $argv[1] ?? 5;
$ratio = $argv[2] ?? 0.10;
$minSize = $argv[3] ?? 3;
$maxDepth = $argv[4] ?? 100;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using a Decision Forest         ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')->setDelimiter(',');

$dataset = new SupervisedDataset(iterator_to_array($dataset));

list ($training, $testing) = $dataset->randomize()->split(0.5);

$pipeline = new Prototype(
    new DecisionForest($trees, $ratio, $minSize, $maxDepth),
    [new OneHotEncoder()],
    [new Accuracy(), new Speed()]
);

echo 'Training a Decision Forest of ' . $trees . ' trees ... ';

$start = microtime(true);

$pipeline->train($training->samples(), $training->outcomes());

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Testing model ...' . "\n";

$pipeline->test($testing->samples(), $testing->outcomes());
