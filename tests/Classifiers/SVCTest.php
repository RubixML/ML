<?php

namespace Rubix\ML\Tests\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Classifiers
 * @requires extension svm
 * @covers \Rubix\ML\Classifiers\SVC
 */
class SVCTest extends TestCase
{
    /**
     * The number of samples in the training set.
     *
     * @var int
     */
    protected const TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 256;

    /**
     * The minimum validation score required to pass the test.
     *
     * @var float
     */
    protected const MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var SVC
     */
    protected $estimator;

    /**
     * @var FBeta
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'male' => new Blob([69.2, 195.7, 40.0], [2.0, 6.0, 0.6]),
            'female' => new Blob([63.7, 168.5, 38.1], [1.6, 5.0, 0.8]),
        ], [0.45, 0.55]);

        $this->estimator = new SVC(1.0, new RBF(), true, 1e-3);

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    /**
     * @after
     */
    protected function tearDown() : void
    {
        if (file_exists('svc.model')) {
            unlink('svc.model');
        }

        if (file_exists('svc.model.classes.json')) {
            unlink('svc.model.classes.json');
        }

        if (is_dir('svc_fail_dir')) {
            chmod('svc_fail_dir', 0755);

            $files = glob('svc_fail_dir/*');

            foreach ($files === false ? [] : $files as $file) {
                chmod($file, 0644);
                unlink($file);
            }

            rmdir('svc_fail_dir');
        }
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SVC::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'c' => 1.0,
            'kernel' => new RBF(),
            'shrinking' => true,
            'tolerance' => 1e-3,
            'cache size' => 100.0,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainSaveLoadPredict() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $this->estimator->train($dataset);

        $this->assertTrue($this->estimator->trained());

        $this->estimator->save('svc.model');

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        $estimator->load('svc.model');

        $predictions = $estimator->predict($testing);

        $expectedClasses = ['male', 'female'];

        foreach ($predictions as $prediction) {
            $this->assertIsString($prediction);
            $this->assertNotSame('', $prediction);
            $this->assertContains($prediction, $expectedClasses);
        }

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    /**
     * @test
     */
    public function saveOverwritesPreviousPair() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

        $this->estimator->save('svc.model');

        $otherGenerator = new Agglomerate([
            'cat' => new Blob([69.2, 195.7, 40.0], [2.0, 6.0, 0.6]),
            'dog' => new Blob([63.7, 168.5, 38.1], [1.6, 5.0, 0.8]),
        ], [0.45, 0.55]);

        $otherDataset = $otherGenerator->generate(self::TRAIN_SIZE);

        $otherDataset->apply(new ZScaleStandardizer());

        $otherEstimator = new SVC(1.0, new RBF(), true, 1e-3);

        $otherEstimator->train($otherDataset);

        $otherEstimator->save('svc.model');

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        $estimator->load('svc.model');

        $predictions = $estimator->predict($otherDataset);

        foreach ($predictions as $prediction) {
            $this->assertContains($prediction, ['cat', 'dog']);
        }

        $data = file_get_contents('svc.model.classes.json');

        $this->assertNotSame(false, $data);

        $this->assertSame(['cat', 'dog'], json_decode($data, true));
    }

    /**
     * @test
     */
    public function failedSaveLeavesExistingPairUntouched() : void
    {
        if (function_exists('posix_geteuid') and posix_geteuid() === 0) {
            $this->markTestSkipped('Permission failures cannot be simulated as root.');
        }

        $dir = 'svc_fail_dir';

        if (!is_dir($dir) and !mkdir($dir) and !is_dir($dir)) {
            $this->fail('Could not create the fixture directory.');
        }

        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

        $this->estimator->save("$dir/svc.model");

        $before = (new SVC(1.0, new RBF(), true, 1e-3));

        $before->load("$dir/svc.model");

        $beforePredictions = $before->predict(Unlabeled::quick($dataset->samples()));

        $otherGenerator = new Agglomerate([
            'cat' => new Blob([69.2, 195.7, 40.0], [2.0, 6.0, 0.6]),
            'dog' => new Blob([63.7, 168.5, 38.1], [1.6, 5.0, 0.8]),
        ], [0.45, 0.55]);

        $otherDataset = $otherGenerator->generate(self::TRAIN_SIZE);

        $otherDataset->apply(new ZScaleStandardizer());

        $otherEstimator = new SVC(1.0, new RBF(), true, 1e-3);

        $otherEstimator->train($otherDataset);

        chmod($dir, 0555);

        try {
            $otherEstimator->save("$dir/svc.model");

            $this->fail('Expected a failed save to throw.');
        } catch (RuntimeException $exception) {
            $this->assertStringContainsString('writable', strtolower($exception->getMessage()));
        } finally {
            chmod($dir, 0755);
        }

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        $estimator->load("$dir/svc.model");

        $this->assertEquals(
            $beforePredictions,
            $estimator->predict(Unlabeled::quick($dataset->samples()))
        );

        $leftovers = glob("$dir/tmp*");

        $this->assertCount(0, $leftovers === false ? [] : $leftovers);
    }

    /**
     * @test
     */
    public function loadWithoutClassMap() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

        $this->estimator->save('svc.model');

        if (file_exists('svc.model.classes.json')) {
            unlink('svc.model.classes.json');
        }

        $this->expectException(RuntimeException::class);

        (new SVC(1.0, new RBF(), true, 1e-3))->load('svc.model');
    }

    /**
     * @test
     */
    public function loadFailureLeavesEstimatorUntouched() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

        $this->estimator->save('svc.model');

        $otherGenerator = new Agglomerate([
            'cat' => new Blob([69.2, 195.7, 40.0], [2.0, 6.0, 0.6]),
            'dog' => new Blob([63.7, 168.5, 38.1], [1.6, 5.0, 0.8]),
        ], [0.45, 0.55]);

        $otherDataset = $otherGenerator->generate(self::TRAIN_SIZE);

        $otherDataset->apply(new ZScaleStandardizer());

        $otherEstimator = new SVC(1.0, new RBF(), true, 1e-3);

        $otherEstimator->train($otherDataset);

        $before = $otherEstimator->predict($otherDataset);

        foreach ($before as $prediction) {
            $this->assertContains($prediction, ['cat', 'dog']);
        }

        unlink('svc.model');

        try {
            $otherEstimator->load('svc.model');

            $this->fail('Expected the load of a missing model to throw.');
        } catch (\Throwable $exception) {
            $this->assertInstanceOf(\svmexception::class, $exception);
        }

        $this->assertTrue($otherEstimator->trained());

        foreach ($otherEstimator->predict($otherDataset) as $prediction) {
            $this->assertContains($prediction, ['cat', 'dog']);
        }

        $this->assertEquals($before, $otherEstimator->predict($otherDataset));
    }

    /**
     * @test
     */
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick([['bad']]));
    }

    /**
     * @test
     */
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick([[1.5]]));
    }
}
