<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\AdaBoost;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\NaiveBayes;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Autism Screener using Boosted Naive Bayes           ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/autism.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new Prototype(new Pipeline(new AdaBoost(NaiveBayes::class, [], 20), [
    new MissingDataImputer('?'),
    new ZScaleStandardizer(),

]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);
