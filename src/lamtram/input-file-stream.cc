// $Id$

// This file is originally from moses with the following copyright info:
/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (C) 2006 University of Edinburgh

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#include <lamtram/macros.h>
#include <lamtram/input-file-stream.h>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <iostream>

using namespace std;
using namespace boost;
using namespace boost::iostreams;

namespace lamtram {

inline bool FileExists(const std::string& filePath)
{
  ifstream ifs(filePath.c_str());
  return !ifs.fail();
}

InputFileStream::InputFileStream(std::string filePath)
  : std::istream(NULL), m_streambuf(NULL), ifs_(NULL) {
    if(!FileExists(filePath)) {
        if(FileExists(filePath+".gz"))
            filePath += ".gz";
        else
            THROW_ERROR("Could not find translation model: " << filePath);
    }
    if(filePath.size() > 3 &&
       filePath.substr(filePath.size() - 3, 3) == ".gz") {
      filtering_streambuf<input> * in = new filtering_streambuf<input>;
      ifs_ = new ifstream(filePath.c_str());
      in->push(gzip_decompressor());
      in->push(*ifs_);
      m_streambuf = in;
    } else {
      std::filebuf* fb = new std::filebuf();
      fb = fb->open(filePath.c_str(), std::ios::in);
      if (! fb) {
        cerr << "Can't read " << filePath.c_str() << endl;
        exit(1);
      }
      m_streambuf = fb;
    }
    this->init(m_streambuf);
}

InputFileStream::~InputFileStream()
{
  delete m_streambuf;
  m_streambuf = NULL;
  if(ifs_) delete ifs_;
}

void InputFileStream::Close()
{
}

}

