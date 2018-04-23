<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\CART;
use Rubix\Engine\AdaBoost;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\MCC;
use Rubix\Engine\Tests\F1Score;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Tests\Informedness;
use Rubix\Engine\Transformers\OneHotEncoder;
use League\Csv\Reader;

$minSize = $argv[1] ?? 1;
$maxDepth = $argv[2] ?? 3;
$experts = $argv[3] ?? 100;
$ratio = $argv[4] ?? 0.10;
$threshold = $argv[5] ?? 0.99;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Breast Cancer Detector using Adaboost               ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/breast-cancer.csv')->setDelimiter(',');

$dataset = SupervisedDataset::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.3);

$prototype = new Prototype(
    new Pipeline(new AdaBoost(CART::class, [$minSize, $maxDepth], $experts, $ratio, $threshold), [new OneHotEncoder()]),
    [new Accuracy(), new F1Score(), new MCC(), new Informedness()]
);

$prototype->train($training);

$prototype->test($testing);
