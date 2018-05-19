<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\AdaBoost;
use Rubix\Engine\Estimators\NaiveBayes;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\Estimators\Wrappers\Pipeline;
use Rubix\Engine\ModelSelection\CrossValidator;
use Rubix\Engine\ModelSelection\ReportGenerator;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\ModelSelection\Reports\AggregateReport;
use Rubix\Engine\ModelSelection\Reports\ConfusionMatrix;
use Rubix\Engine\ModelSelection\Reports\ClassificationReport;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Autism Screener using Boosted Naive Bayes           ║' . "\n";
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

$dataset = new Supervised($samples, $labels);

$estimator = new Pipeline(new AdaBoost(NaiveBayes::class, [], 20, 0.1), [
    new MissingDataImputer('?'),
    new NumericStringConverter(),
]);

$validator = new CrossValidator(new MCC());

$report = new AggregateReport([
    new ConfusionMatrix(),
    new ClassificationReport(),
]);

var_dump($validator->validate($estimator, $dataset->stratifiedFold(10)));

list($training, $testing) = $dataset->stratifiedSplit(0.8);

$estimator->train($training);

$predictions = $estimator->predict($testing);

var_dump($report->generate($predictions, $testing->labels()));
