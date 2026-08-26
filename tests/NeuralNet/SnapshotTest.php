<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Snapshot;
use function sys_get_temp_dir;
use function glob;
use function rmdir;
use function is_dir;
use function unserialize;
use function file_get_contents;

#[Group('NeuralNet')]
#[CoversClass(Snapshot::class)]
class SnapshotTest extends TestCase
{
    protected string $testDir;

    protected function setUp() : void
    {
        $this->testDir = sys_get_temp_dir() . '/rubix-ml-test-snapshots-' . uniqid('', true);

        mkdir($this->testDir, 0o755, true);
    }

    protected function tearDown() : void
    {
        if (is_dir($this->testDir)) {
            $files = glob($this->testDir . '/*/*');

            foreach ($files as $file) {
                @unlink($file);
            }

            $dirs = glob($this->testDir . '/*');

            foreach ($dirs as $dir) {
                @rmdir($dir);
            }

            @rmdir($this->testDir);
        }
    }

    public function testConstructorThrowsWithWrongParameters() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Number of layers and file paths must be equal.');

        new Snapshot(
            layers: [new Dense(1)],
            files: [],
            directory: $this->testDir,
        );
    }

    public function testConstructorThrowsWithNonexistentDirectory() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('does not exist');

        new Snapshot(
            layers: [],
            files: [],
            directory: '/nonexistent/directory',
        );
    }

    public function testTake() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testDir);

        $snapshotDirs = glob($this->testDir . '/*');

        $this->assertCount(1, $snapshotDirs);
        $this->assertTrue(is_dir($snapshotDirs[0]));

        $files = glob($snapshotDirs[0] . '/*.params');

        $this->assertCount(3, $files);

        $snapshot->clean();
    }

    public function testRestore() : void
    {
        $network = $this->createNetwork();

        $originalData = $this->captureNetworkData($network);

        $snapshot = Snapshot::take($network, $this->testDir);

        $snapshotDirs = glob($this->testDir . '/*');
        $files = glob($snapshotDirs[0] . '/*.params');

        foreach ($files as $file) {
            $contents = file_get_contents($file);

            $this->assertNotFalse($contents);

            $data = unserialize($contents);

            $this->assertIsArray($data);

            foreach ($data as $param) {
                $this->assertInstanceOf(\NDArray::class, $param->param());
            }
        }

        $snapshot->restore();

        $restoredData = $this->captureNetworkData($network);

        $this->assertEquals($originalData, $restoredData);

        $snapshot->clean();
    }

    public function testClean() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testDir);

        $snapshotDirs = glob($this->testDir . '/*');

        $this->assertCount(1, $snapshotDirs);

        $snapshot->clean();

        $files = glob($snapshotDirs[0] . '/*');

        $this->assertCount(0, $files);
    }

    /**
     * Create a test network.
     */
    protected function createNetwork() : FeedForward
    {
        $network = new FeedForward(
            input: new Placeholder1D(1),
            hidden: [
                new Dense(10),
                new Activation(new ELU()),
                new Dense(5),
                new Activation(new ELU()),
                new Dense(1),
            ],
            output: new Binary(
                classes: ['yes', 'no'],
                costFn: new CrossEntropy()
            ),
            optimizer: new Stochastic()
        );

        $network->initialize();

        return $network;
    }

    /**
     * Capture parameter data from all parametric layers for comparison.
     *
     * @param Network $network
     * @return list<array<string, array>>
     */
    protected function captureNetworkData(Network $network) : array
    {
        $data = [];

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                $layerData = [];

                foreach ($layer->parameters() as $key => $parameter) {
                    $layerData[$key] = $parameter->param()->toArray();
                }

                $data[] = $layerData;
            }
        }

        return $data;
    }
}
