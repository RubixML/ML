<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Classifiers\AdaBoost;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Classifiers\NaiveBayes;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\Transformers\OneHotEncoder;
use Rubix\Engine\Transformers\MinMaxNormalizer;
use Rubix\Engine\CrossValidation\ReportGenerator;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\AggregateReport;
use Rubix\Engine\CrossValidation\Reports\ConfusionMatrix;
use Rubix\Engine\CrossValidation\Reports\ClassificationReport;
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

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new NaiveBayes(), [
    new NumericStringConverter(),
    new MissingDataImputer('?'),
]);

$validator = new KFold(new MCC(), 10);

$report = new ReportGenerator(new AggregateReport([
    new ConfusionMatrix(),
    new ClassificationReport(),
]), 0.2);

var_dump($validator->score($estimator, $dataset));

var_dump($report->generate($estimator, $dataset));

var_dump($estimator->proba($dataset->randomize()->head(5)));
