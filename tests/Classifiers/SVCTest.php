<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\DataType;
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
use Rubix\ML\Exceptions\JSONException;
use PHPUnit\Framework\TestCase;

#[Group('Classifiers')]
#[RequiresPhpExtension('svm')]
#[CoversClass(SVC::class)]
class SVCTest extends TestCase
{
    /**
     * The number of samples in the training set.
     */
    protected const int TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 256;

    /**
     * The minimum validation score required to pass the test.
     */
    protected const float MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected SVC $estimator;

    protected FBeta $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'male' => new Blob(
                    center: [69.2, 195.7, 40.0],
                    stdDev: [2.0, 6.0, 0.6]
                ),
                'female' => new Blob(
                    center: [63.7, 168.5, 38.1],
                    stdDev: [1.6, 5.0, 0.8]
                ),
            ],
            weights: [0.45, 0.55]
        );

        $this->estimator = new SVC(
            c: 1.0,
            kernel: new RBF(),
            shrinking: true,
            tolerance: 1e-3
        );

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    protected function tearDown() : void
    {
        if (file_exists('svc.model')) {
            unlink('svc.model');
        }

        if (file_exists('svc.model.classes.json')) {
            unlink('svc.model.classes.json');
        }

        if (is_dir('svc_fail_dir')) {
            chmod('svc_fail_dir', 0o755);

            $files = glob('svc_fail_dir/*');

            foreach ($files === false ? [] : $files as $file) {
                chmod($file, 0o644);
                unlink($file);
            }

            rmdir('svc_fail_dir');
        }
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
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

    public function testTrainSaveLoadPredict() : void
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

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function testSaveOverwritesPreviousPair() : void
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

    public function testSaveWithInvalidUTF8LabelThrewBeforeWriting() : void
    {
        $badLabel = "caf\xE9";

        $samples = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        $labels = [
            'ok',
            'ok',
            $badLabel,
            $badLabel,
        ];

        $dataset = Labeled::quick($samples, $labels);

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        $estimator->train($dataset);

        $path = 'svc_invalid_utf8.model';
        $sidecar = "$path.classes.json";

        $this->assertFileDoesNotExist($path);
        $this->assertFileDoesNotExist($sidecar);

        $this->expectException(JSONException::class);

        $estimator->save($path);

        $this->assertFileDoesNotExist($path);
        $this->assertFileDoesNotExist($sidecar);
    }

    public function testSaveFailedWithInvalidUTF8DidNotClobberExistingPair() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

        $this->estimator->save('svc.model');

        $validSidecar = file_get_contents('svc.model.classes.json');

        $this->assertNotSame(false, $validSidecar);

        $badLabel = "caf\xE9";

        $badDataset = Labeled::quick(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            ['ok', 'ok', $badLabel, $badLabel]
        );

        $badEstimator = new SVC(1.0, new RBF(), true, 1e-3);

        $badEstimator->train($badDataset);

        try {
            $badEstimator->save('svc.model');

            $this->fail('Expected saving a model with invalid UTF-8 labels to throw.');
        } catch (JSONException $exception) {
            $this->assertStringContainsString('UTF-8', $exception->getMessage());
        }

        $this->assertSame($validSidecar, file_get_contents('svc.model.classes.json'));

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        $estimator->load('svc.model');

        $predictions = $estimator->predict($dataset);

        foreach ($predictions as $prediction) {
            $this->assertContains($prediction, ['male', 'female']);
        }
    }

    public function testLoadEmptySidecarThrew() : void
    {
        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE));

        $this->estimator->save('svc.model');

        file_put_contents('svc.model.classes.json', '');

        $this->expectException(JSONException::class);

        (new SVC(1.0, new RBF(), true, 1e-3))->load('svc.model');
    }

    public function testLoadSidecarWithTooManyClassesThrows() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

        $this->estimator->save('svc.model');

        file_put_contents('svc.model.classes.json', '["male", "female", "extra"]');

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        try {
            $estimator->load('svc.model');

            $this->fail('Expected the load of a mismatched class map to throw.');
        } catch (RuntimeException $exception) {
            $this->assertStringContainsString('class label map', $exception->getMessage());
        }

        $this->assertFalse($estimator->trained());

        $this->estimator->save('svc.model');

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        $estimator->load('svc.model');

        $predictions = $estimator->predict(Unlabeled::quick($dataset->samples()));

        foreach ($predictions as $prediction) {
            $this->assertContains($prediction, ['male', 'female']);
        }
    }

    public function testLoadSidecarWithTooFewClassesThrows() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

        $this->estimator->save('svc.model');

        file_put_contents('svc.model.classes.json', '["male"]');

        $estimator = new SVC(1.0, new RBF(), true, 1e-3);

        try {
            $estimator->load('svc.model');

            $this->fail('Expected the load of a mismatched class map to throw.');
        } catch (RuntimeException $exception) {
            $this->assertStringContainsString('class label map', $exception->getMessage());
        }

        $this->assertFalse($estimator->trained());
    }

    public function testFailedSaveLeavesExistingPairUntouched() : void
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

        chmod($dir, 0o555);

        try {
            $otherEstimator->save("$dir/svc.model");

            $this->fail('Expected a failed save to throw.');
        } catch (RuntimeException $exception) {
            $this->assertStringContainsString('writable', strtolower($exception->getMessage()));
        } finally {
            chmod($dir, 0o755);
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

    public function testLoadWithoutClassMap() : void
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

    public function testLoadFailureLeavesEstimatorUntouched() : void
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $this->estimator->train($dataset);

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

        $otherEstimator->save('svc.model');

        if (file_exists('svc.model')) {
            unlink('svc.model');
        }

        try {
            $otherEstimator->load('svc.model');

            $this->fail('Expected the load of a missing model to throw.');
        } catch (\Throwable $exception) {
            $this->assertInstanceOf(\SVMException::class, $exception);
        }

        $this->assertTrue($otherEstimator->trained());

        foreach ($otherEstimator->predict($otherDataset) as $prediction) {
            $this->assertContains($prediction, ['cat', 'dog']);
        }

        $this->assertEquals($before, $otherEstimator->predict($otherDataset));
    }

    public function testTrainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']]));
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick(samples: [[1.5]]));
    }
}
