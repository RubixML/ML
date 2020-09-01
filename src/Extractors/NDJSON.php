<?php

namespace Rubix\ML\Extractors;

use InvalidArgumentException;
use League\Flysystem\FilesystemInterface;
use Rubix\ML\FilesystemAware;
use Rubix\ML\Other\Traits\FilesystemTrait;
use RuntimeException;
use Generator;

use function is_null;

/**
 * NDJSON
 *
 * NDJSON or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object
 * Notation (JSON) arrays or objects. The format is like a mix of JSON and CSV and has the
 * advantage of retaining data type information and being read into memory incrementally.
 *
 * > **Note:** Empty lines are ignored by the parser.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSON implements Extractor, FilesystemAware
{
    use FilesystemTrait;

    /**
     * The file handle.
     *
     * @var resource
     */
    protected $handle;

    /**
     * @param string $path
     * @param FilesystemInterface|null $filesystem
     */
    public function __construct(string $path, ?FilesystemInterface $filesystem = null)
    {
        if ($filesystem) {
            $this->setFilesystem($filesystem);
        }

        if (!$this->filesystem()->has($path)) {
            throw new InvalidArgumentException("Path $path does not exist.");
        }

        $handle = $this->filesystem()->readStream($path);

        if (!$handle) {
            throw new RuntimeException("Could not open $path.");
        }

        $this->handle = $handle;
    }

    /**
     * Clean up the file pointer.
     */
    public function __destruct()
    {
        fclose($this->handle);
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \RuntimeException
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        rewind($this->handle);

        $line = 0;

        while (!feof($this->handle)) {
            $data = rtrim(fgets($this->handle) ?: '');

            ++$line;

            if (empty($data)) {
                continue 1;
            }

            $record = json_decode($data, true);

            if (is_null($record)) {
                throw new RuntimeException("Malformed JSON on line $line.");
            }

            yield $record;
        }
    }
}
