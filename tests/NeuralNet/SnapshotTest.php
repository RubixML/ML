<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Snapshot;
use function sys_get_temp_dir;
use function is_file;
use function is_dir;
use function dirname;
use function rmdir;
use function strlen;
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

    public function testTake() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testPath);

        $this->assertFileExists($this->testPath);

        $contents = file_get_contents($this->testPath);

        $this->assertNotFalse($contents);
        $this->assertGreaterThan(0, strlen($contents));

        $snapshot->clean();
    }

    public function testRestore() : void
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
                $this->assertInstanceOf(\NDArray::class, $param->param());
            }

            $offset += $length['len'];
        }

        $snapshot->restore();

        $restoredData = $this->captureNetworkData($network);

        $this->assertEquals($originalData, $restoredData);

        $snapshot->clean();
    }

    public function testClean() : void
    {
        $network = $this->createNetwork();

        $snapshot = Snapshot::take($network, $this->testPath);

        $this->assertFileExists($this->testPath);

        $snapshot->clean();

        $this->assertFileDoesNotExist($this->testPath);
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
