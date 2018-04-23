<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Ridge;
use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\RMSError;
use Rubix\Engine\Tests\RSquared;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Tests\StandardError;
use Rubix\Engine\Tests\MeanAbsoluteError;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use League\Csv\Reader;

$alpha = $argv[1] ?? 0.5;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ MPG Estimator using Ridge Regression                ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/auto-mpg.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.20);

$prototype = new Prototype(new Pipeline(new Ridge($alpha), [
    new MissingDataImputer('?'), new ZScaleStandardizer(),
]), [
    new RMSError(), new MeanAbsoluteError(), new StandardError(), new RSquared(),
]);

$prototype->train($training);

$prototype->test($testing);
