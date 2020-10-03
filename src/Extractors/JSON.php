<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Other\Helpers\JSON as JSONHelper;
use Rubix\ML\Storage\Exceptions\StorageException;
use Rubix\ML\Storage\LocalFilesystem;
use Rubix\ML\Storage\Reader;
use Rubix\ML\Storage\ReadProxy;
use Rubix\ML\Storage\Streams\Stream;
use Generator;

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
class JSON implements Extractor
{
    /**
     * @var \Rubix\ML\Storage\Streams\Stream
     */
    protected $stream;

    /**
     * @param string $location
     * @param \Rubix\ML\Storage\Reader|null $storage
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function __construct(string $location, ?Reader $storage = null)
    {
        if (!$storage) {
            $storage = new LocalFilesystem();
        }

        $storage = new ReadProxy($storage);

        if (!$storage->exists($location)) {
            throw new InvalidArgumentException("Location $location does not exist.");
        }

        $this->stream = $storage->read($location, Stream::READ_ONLY);
    }

    /**
     * Clean up the any open file handles.
     */
    public function __destruct()
    {
        try {
            if ($this->stream and $this->stream->open()) {
                $this->stream->close();
            }
        } catch (StorageException $e) {
            //
        }
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        try {
            $data = $this->stream->contents();
        } catch (StorageException $e) {
            throw new RuntimeException($e->getMessage(), $e->getCode(), $e);
        }

        yield from JSONHelper::decode($data);
    }
}
