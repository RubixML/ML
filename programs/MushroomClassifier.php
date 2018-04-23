<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\MCC;
use Rubix\Engine\Tests\F1Score;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\DecisionForest;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Tests\Informedness;
use Rubix\Engine\Transformers\OneHotEncoder;
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

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')->setDelimiter(',');

$dataset = SupervisedDataset::fromIterator($dataset);

list ($training, $testing) = $dataset->randomize()->stratifiedSplit(0.5);

$prototype = new Prototype(
    new Pipeline(new DecisionForest($trees, $ratio, $minSize, $maxDepth), [new OneHotEncoder()]),
    [new Accuracy(), new F1Score(), new MCC(), new Informedness()]
);

$prototype->train($training);

$prototype->test($testing);
