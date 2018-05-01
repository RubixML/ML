<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Adaline;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use Rubix\Engine\Transformers\DensePolynomialExpander;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Detector using Adaline                  ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/banknotes.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);

$estimator = new Prototype(new Pipeline(new Adaline(12, 100, 5, new Adam(0.001), 1e-8), [
    new DensePolynomialExpander(3),
    new ZScaleStandardizer(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

var_dump($estimator->steps());

$estimator->test($testing);

var_dump($estimator->predict($dataset->sample(0)));
