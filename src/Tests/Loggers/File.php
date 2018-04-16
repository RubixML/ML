<?php

namespace Rubix\Engine\Tests\Loggers;

use InvalidArgumentException;
use RuntimeException;

class File implements Logger
{
    const DATE_FORMAT = 'Y-m-d H:i:s';
    const MODE = 'ab+';

    /**
     * The file handle.
     *
     * @var resource
     */
    protected $handle;

    /**
     * @param  string  $path
     * @return void
     */
    public function __construct(string $path)
    {
        $handle = fopen($path, self::MODE);

        if (!isset($handle)) {
            throw new InvalidArgumentException('Could not open the specified file for writing.');
        }

        $this->handle = $handle;
    }

    /**
     * Log the message to the screen.
     *
     * @param  string  $message
     * @return void
     */
    public function log(string $message) : void
    {
        if (fwrite($this->handle, date(self::DATE_FORMAT) . ' - ' . $message) === false) {
            throw new RuntimeException('Could not write to file, check permissions.');
        }
    }

    /**
     * @return void
     */
    public function __destruct()
    {
        if (isset($this->handle)) {
            fclose($this->handle);
        }
    }
}
