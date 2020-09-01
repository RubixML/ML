<?php

namespace Rubix\ML;

use League\Flysystem\FilesystemInterface;

interface FilesystemAware
{
    /**
     * Return a Filesystem instance.
     *
     * @return FilesystemInterface
     */
    public function filesystem() : FilesystemInterface;
}
