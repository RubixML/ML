<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\Exceptions\RuntimeException;

use function is_dir;
use function dirname;
use function mkdir;
use function file_put_contents;
use function file_get_contents;
use function serialize;
use function unserialize;
use function strlen;
use function pack;
use function unpack;
use function is_file;
use function unlink;

/**
 * Snapshot
 *
 * A snapshot represents the state of a neural network at a moment in time. The
 * parameters are streamed to a single file on disk to minimize memory usage
 * during training.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Snapshot
{
    /**
     * The parametric layers of the network.
     *
     * @var Parametric[]
     */
    protected array $layers;

    /**
     * The file path of the snapshot file.
     *
     * @var string
     */
    protected string $file;

    /**
     * Take a snapshot of the network.
     *
     * @param Network $network
     * @param string $path
     * @return Snapshot
     */
    public static function take(Network $network, string $path) : self
    {
        $parent = dirname($path);

        if (!is_dir($parent)) {
            $created = @mkdir($parent, 0o755, true);

            if (!$created) {
                throw new RuntimeException("Could not create snapshot directory $parent.");
            }
        }

        $layers = [];

        $numLayers = 0;

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                ++$numLayers;
            }
        }

        $written = file_put_contents($path, pack('J', $numLayers));

        if ($written === false) {
            throw new RuntimeException("Could not write snapshot header to $path.");
        }

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                $params = [];

                foreach ($layer->parameters() as $key => $parameter) {
                    $params[$key] = clone $parameter;
                }

                $data = serialize($params);

                file_put_contents($path, pack('J', strlen($data)) . $data, FILE_APPEND);

                unset($params, $data);

                $layers[] = $layer;
            }
        }

        return new self($layers, $path);
    }

    /**
     * Class constructor.
     *
     * @param Parametric[] $layers
     * @param string $file
     */
    public function __construct(array $layers, string $file)
    {
        $this->layers = $layers;
        $this->file = $file;
    }

    /**
     * Restore the network parameters from disk.
     */
    public function restore() : void
    {
        $contents = file_get_contents($this->file);

        if ($contents === false) {
            throw new RuntimeException("Could not read snapshot file {$this->file}.");
        }

        $offset = 0;

        $header = unpack('Jcount', substr($contents, $offset, 8));

        if ($header === false) {
            throw new RuntimeException("Could not read snapshot header from {$this->file}.");
        }

        $count = $header['count'];

        $offset += 8;

        for ($i = 0; $i < $count; ++$i) {
            $length = unpack('Jlen', substr($contents, $offset, 8));

            if ($length === false) {
                throw new RuntimeException("Could not read snapshot length from {$this->file}.");
            }

            $offset += 8;

            $params = unserialize(substr($contents, $offset, $length['len']), ['allowed_classes' => [\Rubix\ML\NeuralNet\Parameter::class, \NDArray::class]]);

            if (!is_array($params)) {
                throw new RuntimeException("Could not unserialize snapshot data from {$this->file}.");
            }

            $offset += $length['len'];

            $layer = $this->layers[$i];

            $layer->restore($params);
        }
    }

    /**
     * Remove the snapshot file from disk.
     */
    public function clean() : void
    {
        if (is_file($this->file)) {
            @unlink($this->file);
        }
    }
}
