<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Adaline;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\DummyEstimator;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use Rubix\Engine\Transformers\DensePolynomialExpander;
use Rubix\Engine\Transformers\Strategies\PopularityContest;
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

$dummy = new Prototype(new DummyEstimator(new PopularityContest()), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$dummy->train($training);
$estimator->train($training);

$dummy->test($testing);
$estimator->test($testing);

echo "\n";

var_dump($estimator->predict($dataset->sample(0)));
