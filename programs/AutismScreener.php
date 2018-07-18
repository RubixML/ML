<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\ExtraTree;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Autism Screener using Boosted Extra Tree            ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/autism.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
    'ethnicity', 'jundice', 'austim', 'country_of_res', 'used_app_before',
    'result', 'age_desc', 'relation',
]));

$labels = iterator_to_array($reader->fetchColumn('diagnosis'));

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new AdaBoost(ExtraTree::class, [10, 5, 1], 100, 0.1, 1e-3), [
    new NumericStringConverter(),
    new MissingDataImputer('?'),
]);

$validator = new KFold(10);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new MulticlassBreakdown(),
    new PredictionSpeed(),
]);

var_dump($validator->test($estimator, $dataset, new MCC()));

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($testing->head(3)));
