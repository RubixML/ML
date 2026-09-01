<?php

namespace Rubix\ML\NeuralNet;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\Exceptions\RuntimeException;

use function is_dir;
use function dirname;
use function mkdir;
use function file_put_contents;
use function fopen;
use function fread;
use function fclose;
use function serialize;
use function unserialize;
use function strlen;
use function pack;
use function unpack;
use function is_file;
use function unlink;
use function iterator_to_array;

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
    public const ALLOWED_CLASSES = [
        Parameter::class,
        Matrix::class,
        Vector::class,
    ];

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

        $layers = [];

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                $parameters = iterator_to_array($layer->parameters());

                $data = serialize($parameters);

                $written = file_put_contents($path, pack('J', strlen($data)) . $data, FILE_APPEND);

                if ($written === false) {
                    throw new RuntimeException("Could not write parameter data to $path.");
                }

                unset($data);

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
        $handle = @fopen($this->file, 'rb');

        if ($handle === false) {
            throw new RuntimeException("Could not read snapshot file {$this->file}.");
        }

        try {
            $header = unpack('Jcount', fread($handle, 8));

            if ($header === false) {
                throw new RuntimeException("Could not read snapshot header from {$this->file}.");
            }

            $count = $header['count'];

            if ($count !== count($this->layers)) {
                throw new RuntimeException(
                    "Snapshot parameter group count $count does not match the " .
                    count($this->layers) . " parametric layers of {$this->file}."
                );
            }

            for ($i = 0; $i < $count; ++$i) {
                $length = unpack('Jlen', fread($handle, 8));

                if ($length === false) {
                    throw new RuntimeException("Could not read snapshot length from {$this->file}.");
                }

                $data = fread($handle, $length['len']);

                if ($data === false || strlen($data) !== $length['len']) {
                    throw new RuntimeException("Could not read snapshot data from {$this->file}.");
                }

                $params = unserialize($data, [
                    'allowed_classes' => self::ALLOWED_CLASSES,
                ]);

                unset($data);

                if (!is_array($params)) {
                    throw new RuntimeException("Could not unserialize snapshot data from {$this->file}.");
                }

                $layer = $this->layers[$i];

                $layer->restore($params);
            }
        } finally {
            fclose($handle);
        }
    }

    /**
     * Remove the snapshot file from disk.
     */
    public function destroy() : void
    {
        if (is_file($this->file)) {
            @unlink($this->file);
        }
    }
}
