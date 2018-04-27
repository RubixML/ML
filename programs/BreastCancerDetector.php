<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\CART;
use Rubix\Engine\AdaBoost;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Transformers\L2Regularizer;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\Reports\ClassificationReport;
use League\Csv\Reader;

$minSize = $argv[1] ?? 1;
$maxDepth = $argv[2] ?? 10;
$experts = $argv[3] ?? 50;
$ratio = $argv[4] ?? 0.10;
$threshold = $argv[5] ?? 0.999;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Breast Cancer Detector using AdaBoost               ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/breast-cancer.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator = new Prototype(new Pipeline(new AdaBoost(CART::class, [$minSize, $maxDepth], $experts, $ratio, $threshold), [
    new MissingDataImputer('?'), new L2Regularizer(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);
