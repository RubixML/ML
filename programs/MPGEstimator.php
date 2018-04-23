<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\LeastSquares;
use Rubix\Engine\Tests\RMSError;
use Rubix\Engine\Tests\RSquared;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Tests\StandardError;
use Rubix\Engine\Tests\MeanAbsoluteError;
use Rubix\Engine\Transformers\L2Regularizer;
use Rubix\Engine\Transformers\MissingDataImputer;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ MPG Estimator using Least Squares                   ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/auto-mpg.csv')->setDelimiter(',')->getRecords();

$dataset = SupervisedDataset::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.20);

$prototype = new Prototype(new Pipeline(new LeastSquares(), [
    new MissingDataImputer('?'), new L2Regularizer(),
]), [
    new RMSError(), new MeanAbsoluteError(), new StandardError(), new RSquared(),
]);

$prototype->train($training);

$prototype->test($testing);
