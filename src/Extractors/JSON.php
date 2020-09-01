<?php

namespace Rubix\ML\Extractors;

use Generator;
use InvalidArgumentException;
use League\Flysystem\FilesystemInterface;
use Rubix\ML\FilesystemAware;
use Rubix\ML\Other\Traits\FilesystemTrait;
use RuntimeException;

use function is_null;

/**
 * JSON
 *
 * Javascript Object Notation is a standardized lightweight plain-text representation that
 * is widely used. JSON has the advantage of retaining type information, however since the
 * entire JSON blob is read on load, it is not cursorable like CSV or NDJSON.
 *
 * References:
 * [1] T. Bray. (2014). The JavaScript Object Notation (JSON) Data Interchange Format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class JSON implements Extractor, FilesystemAware
{
    use FilesystemTrait;

    /**
     * The path to the JSON file.
     *
     * @var string
     */
    protected $path;

    /**
     * @param string $path
     * @param ?FilesystemInterface $filesystem
     * @throws \InvalidArgumentException
     */
    public function __construct(string $path, ?FilesystemInterface $filesystem = null)
    {
        if ($filesystem) {
            $this->setFilesystem($filesystem);
        }

        if (!$this->filesystem()->has($path)) {
            throw new InvalidArgumentException("Path $path does not exist.");
        }

        $this->path = $path;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \RuntimeException
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        $data = $this->filesystem()->read($this->path);

        if (!$data) {
            throw new RuntimeException("Could not open $this->path.");
        }

        $records = json_decode($data, true);

        if (is_null($records)) {
            throw new RuntimeException('Malformed JSON document.');
        }

        yield from $records;
    }
}
