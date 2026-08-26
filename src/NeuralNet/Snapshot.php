<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use function is_dir;
use function mkdir;
use function file_put_contents;
use function file_get_contents;
use function serialize;
use function unserialize;
use function is_file;
use function unlink;
use function rmdir;
use function uniqid;

/**
 * Snapshot
 *
 * A snapshot represents the state of a neural network at a moment in time. The
 * parameters are stored on disk to minimize memory usage during training.
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
     * The default directory for storing snapshot files.
     *
     * @var string
     */
    public const DEFAULT_DIRECTORY = '/tmp/rubix-ml/snapshots';

    /**
     * The parametric layers of the network.
     *
     * @var Parametric[]
     */
    protected array $layers;

    /**
     * The file paths containing the serialized parameters for each layer.
     *
     * @var list<string>
     */
    protected array $files;

    /**
     * The directory where snapshot files are stored.
     *
     * @var string
     */
    protected string $directory;

    /**
     * Take a snapshot of the network.
     *
     * @param Network $network
     * @param string $directory
     * @return Snapshot
     */
    public static function take(Network $network, string $directory = self::DEFAULT_DIRECTORY) : self
    {
        $snapshotDir = $directory . '/' . uniqid('', true);

        if (!is_dir($snapshotDir)) {
            $created = @mkdir($snapshotDir, 0o755, true);

            if (!$created) {
                throw new RuntimeException("Could not create snapshot directory $snapshotDir.");
            }
        }

        $layers = $files = [];

        $index = 0;

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                $params = [];

                foreach ($layer->parameters() as $key => $parameter) {
                    $params[$key] = clone $parameter;
                }

                $filePath = $snapshotDir . "/{$index}.params";

                file_put_contents($filePath, serialize($params));

                unset($params);

                $layers[] = $layer;
                $files[] = $filePath;

                ++$index;
            }
        }

        return new self($layers, $files, $snapshotDir);
    }

    /**
     * Class constructor.
     *
     * @param Parametric[] $layers
     * @param list<string> $files
     * @param string $directory
     * @throws InvalidArgumentException
     */
    public function __construct(array $layers, array $files, string $directory)
    {
        if (count($layers) !== count($files)) {
            throw new InvalidArgumentException('Number of layers and file paths must be equal.');
        }

        if (!is_dir($directory)) {
            throw new InvalidArgumentException("Snapshot directory $directory does not exist.");
        }

        $this->layers = $layers;
        $this->files = $files;
        $this->directory = $directory;
    }

    /**
     * Clean up snapshot files when the object is destroyed.
     */
    public function __destruct()
    {
        $this->clean();
    }

    /**
     * Restore the network parameters from disk.
     */
    public function restore() : void
    {
        foreach ($this->layers as $i => $layer) {
            $filePath = $this->files[$i];

            $contents = file_get_contents($filePath);

            if ($contents === false) {
                throw new RuntimeException("Could not read snapshot file $filePath.");
            }

            $params = unserialize($contents);

            if ($params === false) {
                throw new RuntimeException("Could not unserialize snapshot file $filePath.");
            }

            $layer->restore($params);
        }
    }

    /**
     * Remove the snapshot files from disk.
     */
    public function clean() : void
    {
        foreach ($this->files as $filePath) {
            if (is_file($filePath)) {
                @unlink($filePath);
            }
        }

        if (is_dir($this->directory)) {
            @rmdir($this->directory);
        }
    }
}
