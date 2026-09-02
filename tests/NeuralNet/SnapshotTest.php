<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet;

use Tensor\Tensor;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\Exceptions\RuntimeException;

use function sys_get_temp_dir;
use function is_file;
use function is_dir;
use function dirname;
use function rmdir;
use function strlen;
use function pack;
use function substr;
use function file_put_contents;
use function unpack;

#[Group('NeuralNet')]
#[CoversClass(Snapshot::class)]
class SnapshotTest extends TestCase
{
    protected string $testPath;

    protected function setUp() : void
    {
        $this->testPath = sys_get_temp_dir() . '/rubix-ml-test-' . uniqid('', true) . '/snapshot.dat';
    }

    protected function tearDown() : void
    {
        if (is_file($this->testPath)) {
            @unlink($this->testPath);
        }

        $parent = dirname($this->testPath);

        if (is_dir($parent)) {
            @rmdir($parent);
        }
    }

    #[Test]
    public function take() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testPath);

        $this->assertFileExists($this->testPath);

        $contents = file_get_contents($this->testPath);

        $this->assertNotFalse($contents);
        $this->assertGreaterThan(0, strlen($contents));

        $snapshot->destroy();
    }

    #[Test]
    public function restore() : void
    {
        $network = $this->createNetwork();

        $originalData = $this->captureNetworkData($network);

        $snapshot = Snapshot::take($network, $this->testPath);

        $contents = file_get_contents($this->testPath);

        $this->assertNotFalse($contents);

        $offset = 0;
        $header = unpack('Jcount', substr($contents, $offset, 8));

        $this->assertIsArray($header);
        $this->assertSame(3, $header['count']);

        $offset += 8;

        for ($i = 0; $i < $header['count']; ++$i) {
            $length = unpack('Jlen', substr($contents, $offset, 8));

            $this->assertIsArray($length);
            $offset += 8;

            $params = unserialize(substr($contents, $offset, $length['len']));

            $this->assertIsArray($params);

            foreach ($params as $param) {
                $this->assertInstanceOf(Tensor::class, $param->param());
            }

            $offset += $length['len'];
        }

        $dataset = Labeled::quick(
            [[1.0], [0.5], [2.0]],
            ['yes', 'no', 'yes']
        );

        $network->roundtrip($dataset);

        $mutatedData = $this->captureNetworkData($network);

        $this->assertNotEquals($originalData, $mutatedData);

        $snapshot->restore();

        $restoredData = $this->captureNetworkData($network);

        $this->assertEquals($originalData, $restoredData);

        $snapshot->destroy();
    }

    #[Test]
    public function clean() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testPath);

        $this->assertFileExists($this->testPath);

        $snapshot->destroy();

        $this->assertFileDoesNotExist($this->testPath);
    }

    #[Test]
    public function restoreMismatchedCount() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testPath);

        $contents = file_get_contents($this->testPath);

        $this->assertNotFalse($contents);

        $this->writeCorruptedFile(
            $network,
            pack('J', 99) . substr($contents, 8)
        );

        $this->expectException(RuntimeException::class);

        $snapshot->restore();
    }

    #[Test]
    public function restoreTruncatedData() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testPath);

        $contents = file_get_contents($this->testPath);

        $this->assertNotFalse($contents);

        $numLayers = 0;

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                ++$numLayers;
            }
        }

        $groupLength = unpack('Jlen', substr($contents, 8, 8));

        $this->assertIsArray($groupLength);

        $payload = substr($contents, 16, $groupLength['len']);

        $this->assertNotFalse($payload);

        $truncated = substr($payload, 0, max(1, strlen($payload) - 4));

        $this->writeCorruptedFile(
            $network,
            pack('J', $numLayers) . pack('J', strlen($payload) + 4) . $truncated
        );

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Could not read snapshot data');

        $snapshot->restore();
    }

    #[Test]
    public function restoreIsAtomic() : void
    {
        $network = $this->createNetwork();

        $originalData = $this->captureNetworkData($network);

        $snapshot = Snapshot::take($network, $this->testPath);

        $contents = file_get_contents($this->testPath);

        $this->assertNotFalse($contents);

        $this->writeCorruptedFile(
            $network,
            pack('J', 99) . substr($contents, 8)
        );

        $this->expectException(RuntimeException::class);

        try {
            $snapshot->restore();
        } finally {
            $this->assertEquals($originalData, $this->captureNetworkData($network));
        }
    }

    /**
     * Build a snapshot object bound to the network's layers at the corrupt path.
     *
     * @param Network $network
     * @param string $contents
     * @return Snapshot
     */
    protected function writeCorruptedFile(Network $network, string $contents) : Snapshot
    {
        file_put_contents($this->testPath, $contents);

        $layers = [];

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                $layers[] = $layer;
            }
        }

        return new Snapshot($layers, $this->testPath);
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
                costFn:  new BinaryCrossEntropy()
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
                    $layerData[$key] = $parameter->param()->asArray();
                }

                $data[] = $layerData;
            }
        }

        return $data;
    }
}
